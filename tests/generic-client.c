/*
 * Copyright © 2008-2014 Stéphane Raimbault <stephane.raimbault@gmail.com>
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#include <modbus.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "generic-test.h"

/* The goal of this program is to provide a diagnostic client that permits a
   user to manually exercise the following major functions of libmodbus:

   - write_coil
   - read_bits
   - write_coils
   - write_register
   - read_registers
   - write_registers
   - read_registers

   In addition, the program can be called with random values, such that the
   request type and request contents are randomly generated.
*/

// Globals and macros used only in this file
static int verbose_mode = FALSE;
static int randomized_mode = 0;

#define GENERIC_ADDR_MAX                      0xFFFF
#define GENERIC_READ_COILS_MAX                2000
#define GENERIC_WRITE_COILS_MAX               0x07B0   // 1968
#define GENERIC_READ_REGISTERS_MAX            0x07D    // 125
#define GENERIC_WRITE_REGISTERS_MAX           0x07B    // 123
#define GENERIC_WRITEREAD_REGISTERS_READ_MAX  0x07D    // 125
#define GENERIC_WRITEREAD_REGISTERS_WRITE_MAX 0x079    // 121
#define GENERIC_REGISTER_VAL_MAX              0xFFFF

// The following have been moved to generic-test.h
// #define SERVER_ID       17
// #define ADDRESS_START    0
// #define ADDRESS_END     99


// Data structures used only in this file
enum REQUEST_TYPE {
    READ_COILS,
    WRITE_SINGLE_COIL,
    WRITE_COILS,
    READ_REGISTERS,
    WRITE_SINGLE_REGISTER,
    WRITE_REGISTERS,
    WRITEREAD_REGISTERS,
    NUM_REQUEST_TYPES,
    UNKNOWN_REQUEST_TYPE
};

typedef struct request_read_coils {
    int addr; // hex address (0x0000 - 0xFFFF)
    int num;  // decimal count of bits (1 - 2000)
} r_coils_t;

typedef struct request_write_single_coil {
    int addr; // hex address (0x0000 - 0xFFFF)
    int bit;  // 0 or 1
} w_coil_t;

typedef struct request_write_coils {
    int addr; // hex address (0x0000 - 0xFFFF)
    int num;  // decimal count of bits (1 - 1968) # max 0x07B0
    uint8_t bits[GENERIC_WRITE_COILS_MAX];
} w_coils_t;

typedef struct request_read_registers {
    int addr; // hex address (0x0000 - 0xFFFF)
    int num;  // decimal count of registers (1 - 125) # max 0x07D
} r_registers_t;

typedef struct request_write_single_register {
    int addr;     // hex address (0x0000 - 0xFFFF)
    uint16_t val; // 2-byte hex value (0x0000 - 0xFFFF)
} w_register_t;

typedef struct request_write_registers {
    int addr; // hex address (0x0000 - 0xFFFF)
    int num;  // decimal count of registers (1 - 123) # max 0x07B
    uint16_t vals[GENERIC_WRITE_REGISTERS_MAX];
} w_registers_t;

typedef struct request_writeread_registers { // NOTE: write occurs before read
    int r_addr; // hex address (0x0000 - 0xFFFF)
    int r_num;  // decimal count of registers to read (1 - 125) # max 0x07D
    int w_addr; // hex address (0x0000 - 0xFFFF)
    int w_num;  // decimal count of registers to write (1 - 121) # max 0x079
    uint16_t vals[GENERIC_WRITEREAD_REGISTERS_WRITE_MAX];
} wr_registers_t;

typedef struct request_generic {
    enum REQUEST_TYPE type;
    union request_u {
        r_coils_t r_coils;
        w_coil_t w_coil;
        w_coils_t w_coils;
        r_registers_t r_regs;
        w_register_t w_reg;
        w_registers_t w_regs;
        wr_registers_t wr_regs;
    } u;
} generic_req_t;

// Prototypes for functions used only in this file
static void usage(const char *progname);
static int selftest();
static void onetest(const char *name, int predicate, int *pass_count,
                    int *fail_count);
static int parse_r_coils(const char *line, generic_req_t *request);
static int parse_w_coil(const char *line, generic_req_t *request);
static int parse_w_coils(const char *line, generic_req_t *request);
static int parse_r_registers(const char *line, generic_req_t *request);
static int parse_w_register(const char *line, generic_req_t *request);
static int parse_w_registers(const char *line, generic_req_t *request);
static int parse_wr_registers(const char *line, generic_req_t *request);
static int parse_generic_req(const char *line, generic_req_t *request);
static generic_req_t *next_request(void);
static enum REQUEST_TYPE get_request_type(const char *line);
static int startswith(const char *s, const char *pattern);
static char** splitline(char *line, int *num_tokens); // modifies line (strtok)
static int splitlinetest(const char *line, int expected);
static int randrange(int M, int N);
static int string_to_int(const char *s);

////////////////////////////////////////////////////////////////////////////

static void usage(const char *progname) {
    fprintf(stderr, "Usage: %s [serverIP [port]]\n", progname);
    fprintf(stderr, "Generic modbus client controlled by text requests.\n\n");

    const char *option_text =
        "  -r, --record F       record network activity in ktest file F\n"
        "  -p, --playback F     play network activity from ktest file F\n"
        "  -R, --random         choose random modbus messages to send\n"
        "  -s, --selftest       run self-test and exit\n"
        "  -v, --verbose        be more chatty on stderr\n"
        "  -h, --help           display this message and exit\n";

    const char *request_syntax =
        "  read_coils <addr> <num>\n"
        "    addr    = hex address (0x0000 - 0xFFFF)\n"
        "    num     = decimal count of bits (1 - 2000)\n"
        "\n"
        "  write_single_coil <addr> <bit>\n"
        "    addr    = hex address (0x0000 - 0xFFFF)\n"
        "    bit     = 0 or 1\n"
        "\n"
        "  write_coils <addr> <num> <bit> [<bit>...]\n"
        "    addr    = hex address (0x0000 - 0xFFFF)\n"
        "    num     = decimal count of bits (1 - 1968) # max 0x07B0\n"
        "    bit     = 0 or 1\n"
        "\n"
        "  read_registers <addr> <num>\n"
        "    addr    = hex address (0x0000 - 0xFFFF)\n"
        "    num     = decimal count of registers (1 - 125) # max 0x07D\n"
        "\n"
        "  write_single_register <addr> <val>\n"
        "    addr    = hex address (0x0000 - 0xFFFF)\n"
        "    val     = 2-byte hex value (0x0000 - 0xFFFF)\n"
        "\n"
        "  write_registers <addr> <num> <val> [<val>...]\n"
        "    addr    = hex address (0x0000 - 0xFFFF)\n"
        "    num     = decimal count of registers (1 - 123) # max 0x07B\n"
        "    val     = 2-byte hex value (0x0000 - 0xFFFF)\n"
        "\n"
        "  writeread_registers <r_addr> <r_num> <w_addr> <w_num> <val> [<val>...]\n"
        "    r_addr  = hex address (0x0000 - 0xFFFF)\n"
        "    r_num   = decimal count of registers (1 - 125) # max 0x07D\n"
        "    w_addr  = hex address (0x0000 - 0xFFFF)\n"
        "    w_num   = decimal count of registers (1 - 121) # max 0x079\n"
        "    val     = 2-byte hex value (0x0000 - 0xFFFF)\n"
        "    NOTE: write occurs before read\n";

    fprintf(stderr, "%s\n", option_text);
    fprintf(stderr, "Request syntax:\n\n%s\n", request_syntax);

    exit(2);
}


/* At each loop, the program works in the range ADDRESS_START to
 * ADDRESS_END then ADDRESS_START + 1 to ADDRESS_END and so on.
 */
int main(int argc, char *argv[])
{
    /* Abbreviations: "nb" = number, "rq" = request, "rp" = reply */
    modbus_t *ctx;
    int rc;
    int nb_fail;
    int addr = 0;
    int nb;
    uint8_t *tab_rq_bits;
    uint8_t *tab_rp_bits;
    uint16_t *tab_rq_registers;
    uint16_t *tab_rw_rq_registers;
    uint16_t *tab_rp_registers;

    // Command line option flags / storage
    int record_selected = 0;
    int playback_selected = 0;
    int run_self_test = 0;
    const char *ktest_filename = NULL;
    const char *server_address = "127.0.0.1";
    int server_port = 1502;

    // Process command line arguments.
    while (1) {
        int c; // cmd line option char
        static struct option long_options[] = {
            {"record", required_argument, 0, 'r'},
            {"playback", required_argument, 0, 'p'},
            {"random", no_argument, 0, 'R'},
            {"selftest", no_argument, 0, 's'},
            {"verbose", no_argument, 0, 'v'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        // getopt_long stores the option index here.
        int option_index = 0;

        c = getopt_long(argc, argv, "r:p:Rsvh", long_options, &option_index);

        // Detect the end of the options.
        if (c == -1)
            break;

        switch (c) {
        case 0:
            fprintf(stderr, "Option that sets flag: should never happen.\n");
            break;
        case 'r':
            record_selected = 1;
            ktest_filename = optarg;
            break;
        case 'p':
            playback_selected = 1;
            ktest_filename = optarg;
            break;
        case 'R':
            randomized_mode = 1;
            break;
        case 's':
            run_self_test = 1;
            verbose_mode = TRUE;
            break;
        case 'v':
            verbose_mode = TRUE;
            break;
        case 'h':
            usage(argv[0]);
            break;
        case '?':
            // getopt_long already printed an error message.
            usage(argv[0]);
            break;
        default:
            usage(argv[0]);
        }
    }
    if (record_selected && playback_selected) {
        fprintf(stderr, "Error: at most one of record (-r) or playback (-p) may"
                " be selected.\n\n");
        usage(argv[0]);
    }

    /* Process remaining command line arguments (not options): IP and port. */
    if (optind < argc) {
        server_address = argv[optind];
        optind++;
    }
    if (optind < argc) {
        server_port = atoi(argv[optind]);
        optind++;
    }
    if (optind < argc) {
        fprintf(stderr, "Error: too many arguments\n\n");
        usage(argv[0]);
    }

    /* Self-test? */
    if (run_self_test) {
        return selftest();
    }

    /* RTU (unused) */
    /*
    ctx = modbus_new_rtu("/dev/ttyUSB0", 19200, 'N', 8, 1);
    modbus_set_slave(ctx, SERVER_ID);
    */

    /* TCP */
    ctx = modbus_new_tcp(server_address, server_port);
    modbus_set_debug(ctx, verbose_mode);

    if (modbus_connect(ctx) == -1) {
        fprintf(stderr, "Connection failed: %s\n",
                modbus_strerror(errno));
        modbus_free(ctx);
        return -1;
    }

    /* Allocate and initialize the different memory spaces */
    nb = ADDRESS_END - ADDRESS_START;

    tab_rq_bits = (uint8_t *) malloc(nb * sizeof(uint8_t));
    memset(tab_rq_bits, 0, nb * sizeof(uint8_t));

    tab_rp_bits = (uint8_t *) malloc(nb * sizeof(uint8_t));
    memset(tab_rp_bits, 0, nb * sizeof(uint8_t));

    tab_rq_registers = (uint16_t *) malloc(nb * sizeof(uint16_t));
    memset(tab_rq_registers, 0, nb * sizeof(uint16_t));

    tab_rp_registers = (uint16_t *) malloc(nb * sizeof(uint16_t));
    memset(tab_rp_registers, 0, nb * sizeof(uint16_t));

    tab_rw_rq_registers = (uint16_t *) malloc(nb * sizeof(uint16_t));
    memset(tab_rw_rq_registers, 0, nb * sizeof(uint16_t));

    nb_fail = 0;

    char *line = NULL;
    size_t line_len = 0;
    ssize_t line_read = 0;
    while ((line_read = getline(&line, &line_len, stdin)) != -1) {
        int i;

        nb = ADDRESS_END - addr;

        /* WRITE BIT */
        rc = modbus_write_bit(ctx, addr, tab_rq_bits[0]);
        if (rc != 1) {
            printf("ERROR modbus_write_bit (%d)\n", rc);
            printf("Address = %d, value = %d\n", addr, tab_rq_bits[0]);
            nb_fail++;
        } else {
            rc = modbus_read_bits(ctx, addr, 1, tab_rp_bits);
            if (rc != 1 || tab_rq_bits[0] != tab_rp_bits[0]) {
                printf("ERROR modbus_read_bits single (%d)\n", rc);
                printf("address = %d\n", addr);
                nb_fail++;
            }
        }

        /* MULTIPLE BITS */
        rc = modbus_write_bits(ctx, addr, nb, tab_rq_bits);
        if (rc != nb) {
            printf("ERROR modbus_write_bits (%d)\n", rc);
            printf("Address = %d, nb = %d\n", addr, nb);
            nb_fail++;
        } else {
            rc = modbus_read_bits(ctx, addr, nb, tab_rp_bits);
            if (rc != nb) {
                printf("ERROR modbus_read_bits\n");
                printf("Address = %d, nb = %d\n", addr, nb);
                nb_fail++;
            } else {
                for (i=0; i<nb; i++) {
                    if (tab_rp_bits[i] != tab_rq_bits[i]) {
                        printf("ERROR modbus_read_bits\n");
                        printf("Address = %d, value %d (0x%X) != %d (0x%X)\n",
                               addr, tab_rq_bits[i], tab_rq_bits[i],
                               tab_rp_bits[i], tab_rp_bits[i]);
                        nb_fail++;
                    }
                }
            }
        }

        /* SINGLE REGISTER */
        rc = modbus_write_register(ctx, addr, tab_rq_registers[0]);
        if (rc != 1) {
            printf("ERROR modbus_write_register (%d)\n", rc);
            printf("Address = %d, value = %d (0x%X)\n",
                   addr, tab_rq_registers[0], tab_rq_registers[0]);
            nb_fail++;
        } else {
            rc = modbus_read_registers(ctx, addr, 1, tab_rp_registers);
            if (rc != 1) {
                printf("ERROR modbus_read_registers single (%d)\n", rc);
                printf("Address = %d\n", addr);
                nb_fail++;
            } else {
                if (tab_rq_registers[0] != tab_rp_registers[0]) {
                    printf("ERROR modbus_read_registers single\n");
                    printf("Address = %d, value = %d (0x%X) != %d (0x%X)\n",
                           addr, tab_rq_registers[0], tab_rq_registers[0],
                           tab_rp_registers[0], tab_rp_registers[0]);
                    nb_fail++;
                }
            }
        }

        /* MULTIPLE REGISTERS */
        rc = modbus_write_registers(ctx, addr, nb, tab_rq_registers);
        if (rc != nb) {
            printf("ERROR modbus_write_registers (%d)\n", rc);
            printf("Address = %d, nb = %d\n", addr, nb);
            nb_fail++;
        } else {
            rc = modbus_read_registers(ctx, addr, nb, tab_rp_registers);
            if (rc != nb) {
                printf("ERROR modbus_read_registers (%d)\n", rc);
                printf("Address = %d, nb = %d\n", addr, nb);
                nb_fail++;
            } else {
                for (i=0; i<nb; i++) {
                    if (tab_rq_registers[i] != tab_rp_registers[i]) {
                        printf("ERROR modbus_read_registers\n");
                        printf("Address = %d, value %d (0x%X) != %d (0x%X)\n",
                               addr, tab_rq_registers[i], tab_rq_registers[i],
                               tab_rp_registers[i], tab_rp_registers[i]);
                        nb_fail++;
                    }
                }
            }
        }
        /* R/W MULTIPLE REGISTERS */
        rc = modbus_write_and_read_registers(ctx,
                                             addr, nb, tab_rw_rq_registers,
                                             addr, nb, tab_rp_registers);
        if (rc != nb) {
            printf("ERROR modbus_read_and_write_registers (%d)\n", rc);
            printf("Address = %d, nb = %d\n", addr, nb);
            nb_fail++;
        } else {
            for (i=0; i<nb; i++) {
                if (tab_rp_registers[i] != tab_rw_rq_registers[i]) {
                    printf("ERROR modbus_read_and_write_registers READ\n");
                    printf("Address = %d, value %d (0x%X) != %d (0x%X)\n",
                           addr, tab_rp_registers[i], tab_rw_rq_registers[i],
                           tab_rp_registers[i], tab_rw_rq_registers[i]);
                    nb_fail++;
                }
            }

            rc = modbus_read_registers(ctx, addr, nb, tab_rp_registers);
            if (rc != nb) {
                printf("ERROR modbus_read_registers (%d)\n", rc);
                printf("Address = %d, nb = %d\n", addr, nb);
                nb_fail++;
            } else {
                for (i=0; i<nb; i++) {
                    if (tab_rw_rq_registers[i] != tab_rp_registers[i]) {
                        printf("ERROR modbus_read_and_write_registers WRITE\n");
                        printf("Address = %d, value %d (0x%X) != %d (0x%X)\n",
                               addr, tab_rw_rq_registers[i], tab_rw_rq_registers[i],
                               tab_rp_registers[i], tab_rp_registers[i]);
                        nb_fail++;
                    }
                }
            }
        }
    }

    printf("Test: ");
    if (nb_fail)
        printf("%d FAILS\n", nb_fail);
    else
        printf("SUCCESS\n");

    /* Free the memory */
    free(tab_rq_bits);
    free(tab_rp_bits);
    free(tab_rq_registers);
    free(tab_rp_registers);
    free(tab_rw_rq_registers);
    if (line) {
        free(line);
        line = NULL;
    }

    /* Close the connection */
    modbus_close(ctx);
    modbus_free(ctx);

    return 0;
}

static void onetest(const char *name, int predicate, int *pass_count,
                    int *fail_count) {
    int test_num = *pass_count + *fail_count;
    const char *result_string = predicate ? "PASS" : "FAIL";
    if (predicate) {
        (*pass_count)++;
    } else {
        (*fail_count)++;
    }
    printf("Test #%4.4d | %s : %s\n", test_num, name, result_string);
}

static int splitlinetest(const char *line, int expected) {
    char *linecopy = strdup(line);
    int num_tokens = -1;
    char **tokens = splitline(linecopy, &num_tokens);
    free(tokens);
    free(linecopy);
    return (num_tokens == expected);
}

static int splitlinetestcontents(const char *line, int expected,
                                 const char *tokens[]) {
    char *linecopy = strdup(line);
    int num_tokens = -1;
    char **tokens2 = splitline(linecopy, &num_tokens);
    if (num_tokens != expected) {
        free(tokens2);
        free(linecopy);
        return 0;
    }
    int result = 1;
    for (int i = 0; i < expected; i++) {
        if (tokens == NULL || tokens2 == NULL ||
            strcmp(tokens[i], tokens2[i]) != 0) {
            result = 0;
        }
    }
    free(tokens2);
    free(linecopy);
    return result;
}

static int selftest() {
    int pass = 0;
    int fail = 0;
    printf("Running self-tests...\n");

    onetest("startswith1", startswith("Hello world", "Hello"), &pass, &fail);
    onetest("startswith2", !startswith("Bilb", "Bilbo"), &pass, &fail);
    onetest("startswith3", !startswith("bill", "bell"), &pass, &fail);

    onetest("string_to_int1", string_to_int("42") == 42, &pass, &fail);
    onetest("string_to_int2", string_to_int("0xFF") == 0xFF, &pass, &fail);
    onetest("string_to_int3", string_to_int("0X3f") == 0x3F, &pass, & fail);

    onetest("get_request_type1",
            get_request_type("read_coils ") == READ_COILS,
            &pass, &fail);
    onetest("get_request_type2",
            get_request_type("write_single_coil ") == WRITE_SINGLE_COIL,
            &pass, &fail);
    onetest("get_request_type3",
            get_request_type("write_coils ") == WRITE_COILS,
            &pass, &fail);
    onetest("get_request_type4",
            get_request_type("read_registers ") == READ_REGISTERS,
            &pass, &fail);
    onetest("get_request_type5",
            get_request_type("write_single_register ") == WRITE_SINGLE_REGISTER,
            &pass, &fail);
    onetest("get_request_type6",
            get_request_type("write_registers ") == WRITE_REGISTERS,
            &pass, &fail);
    onetest("get_request_type7",
            get_request_type("writeread_registers ") == WRITEREAD_REGISTERS,
            &pass, &fail);
    onetest("get_request_type8",
            get_request_type("writeread_register ") == UNKNOWN_REQUEST_TYPE,
            &pass, &fail);

    onetest("splitline1", splitlinetest("Frodo Baggins", 2), &pass, &fail);
    onetest("splitline2", splitlinetest("  A  B C  ", 3), &pass, &fail);
    onetest("splitline3", !splitlinetest("  A  B C  ", 4), &pass, &fail);
    onetest("splitline4", splitlinetest("  A  B C D", 4), &pass, &fail);
    onetest("splitline5", splitlinetest("", 0), &pass, &fail);
    onetest("splitline6", splitlinetest("   ", 0), &pass, &fail);

    const char *tokens7[] = {"A", "B", "C"};
    onetest("splitline7", splitlinetestcontents(" A  B C   ", 3, tokens7),
            &pass, &fail);

    generic_req_t req;
    int ret = parse_r_coils("read_coils 0x42 38", &req);
    onetest("parse_r_coils1",
            (req.type == READ_COILS) &&
            (req.u.r_coils.addr == 0x42) && (req.u.r_coils.num == 38),
            &pass, &fail);
    ret = parse_w_coil("write_single_coil 0x42 1", &req);
    onetest("parse_w_coil1",
            (req.type == WRITE_SINGLE_COIL) &&
            (req.u.w_coil.addr == 0x42) && (req.u.w_coil.bit == 1),
            &pass, &fail);
    ret = parse_w_coils("write_coils 0x42 3 1 1 0", &req);
    onetest("parse_w_coils1",
            (req.type == WRITE_COILS) &&
            (req.u.w_coils.addr == 0x42) && (req.u.w_coils.num == 3) &&
            (req.u.w_coils.bits[0] == 1) && (req.u.w_coils.bits[1] == 1) &&
            (req.u.w_coils.bits[2] == 0),
            &pass, &fail);
    ret = parse_r_registers("read_registers 0x42 39", &req);
    onetest("parse_r_registers1",
            (req.type == READ_REGISTERS) &&
            (req.u.r_regs.addr == 0x42) && (req.u.r_regs.num == 39),
            &pass, &fail);
    ret = parse_w_register("write_single_register 0x46 0xBEEF", &req);
    onetest("parse_w_register1",
            (req.type == WRITE_SINGLE_REGISTER) &&
            (req.u.w_reg.addr == 0x46) && (req.u.w_reg.val == 0xBEEF),
            &pass, &fail);
    ret = parse_w_registers("write_registers 0x42 3 0xEA75 0xDEAD 0xBEEF",
                            &req);
    onetest("parse_w_registers1",
            (req.type == WRITE_REGISTERS) &&
            (req.u.w_regs.addr == 0x42) && (req.u.w_regs.num == 3) &&
            (req.u.w_regs.vals[0] == 0xEA75) &&
            (req.u.w_regs.vals[1] == 0xDEAD) &&
            (req.u.w_regs.vals[2] == 0xBEEF),
            &pass, &fail);
    ret = parse_wr_registers("writeread_registers 0x44 5 0x55 3 "
                             "0xCAFE 0xBA5E 0xBA11",
                             &req);
    onetest("parse_wr_registers1",
            (req.type == WRITEREAD_REGISTERS) &&
            (req.u.wr_regs.r_addr == 0x44) && (req.u.wr_regs.r_num == 5) &&
            (req.u.wr_regs.w_addr == 0x55) && (req.u.wr_regs.w_num == 3) &&
            (req.u.wr_regs.vals[0] == 0xCAFE) &&
            (req.u.wr_regs.vals[1] == 0xBA5E) &&
            (req.u.wr_regs.vals[2] == 0xBA11),
            &pass, &fail);

    printf("%d self-tests: %d pass, %d fail.\n", pass + fail, pass, fail);
    if (fail == 0) {
        return 0; // success
    } else {
        return 1; // failure
    }
}

// Return a somewhat random integer in the range [M, N] inclusive
static int randrange(int M, int N) {
    assert(0 <= M && M <= N);
    int val = M + rand() % (N - M + 1);
    assert(M <= val && val <= N);
    return val;
}

// Parse either a decimal or hex string (starting with "0x") and return an int.
static int string_to_int(const char *s)
{
    if (strlen(s) >= 2 &&
        (startswith(s, "0x") || startswith(s, "0X"))) {
        return (int) strtol(s, NULL, 16);
    } else {
        return atoi(s);
    }
}


// NOTE: modifies line (strtok), return NULL if failure, or if there are no
// tokens. Allocates memory for the returned array of pointers that the caller
// must eventually free. Note that the pointers themselves simply point into the
// "line" input string, and not to newly allocated memory.  This function writes
// *num_tokens to be the number of tokens seen.
static char** splitline(char *line, int *num_tokens) {
    // capacity available for expandable array
    int capacity = 1;

    // Allocate token list (to be returned)
    char **token_list = (char **) malloc(sizeof(char*) * capacity);

    // Split line into tokens
    int i = 0;
    char *saveptr = NULL;
    for (char *str = line; ; str = NULL, i++) {
        char *token = strtok_r(str, " ", &saveptr);
        if (!token)
            break;
        if (i >= capacity) {
            capacity *= 2;
            token_list = (char **)realloc(token_list, sizeof(char*) * capacity);
        }
        token_list[i] = token;
    }

    // Output: number of tokens seen
    *num_tokens = i;

    // If there are no tokens, return NULL
    if (i == 0) {
        free(token_list);
        return NULL;
    } else {
        return token_list;
    }
}

static int parse_generic_req(const char *line, generic_req_t *request) {
    int ret = -1;
    enum REQUEST_TYPE req_type = get_request_type(line);
    if (req_type == UNKNOWN_REQUEST_TYPE) {
        fprintf(stderr, "Unknown request type: %s", line);
        return -1;
    }
    switch (req_type) {
    case READ_COILS:
        ret = parse_r_coils(line, request);
        break;
    case WRITE_SINGLE_COIL:
        ret = parse_w_coil(line, request);
        break;
    case WRITE_COILS:
        ret = parse_w_coils(line, request);
        break;
    case READ_REGISTERS:
        ret = parse_r_registers(line, request);
        break;
    case WRITE_SINGLE_REGISTER:
        ret = parse_w_register(line, request);
        break;
    case WRITE_REGISTERS:
        ret = parse_w_registers(line, request);
        break;
    case WRITEREAD_REGISTERS:
        ret = parse_wr_registers(line, request);
        break;
    default:
        assert(FALSE); // should never happen
    }
    return ret;
}

static int parse_r_coils(const char *line, generic_req_t *request) {
    char *linecopy = strdup(line);
    const char *TYPE_STRING = "read_coils";
    const int TOKENS_EXPECTED = 3;
    int num_tokens = 0;

    char **tokens = splitline(linecopy, &num_tokens);
    if (!tokens) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(linecopy);
        return -1;
    }
    if (num_tokens != TOKENS_EXPECTED) {
        fprintf(stderr, "Error: expected %d tokens, got %d\n",
                TOKENS_EXPECTED, num_tokens);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // "read_coils"
    if (strcmp(tokens[0], TYPE_STRING) != 0) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // hex address (0x0000 - 0xFFFF)
    int addr = string_to_int(tokens[1]);
    if (addr < 0 || addr > GENERIC_ADDR_MAX ||
        addr < ADDRESS_START || addr > ADDRESS_END) {
        fprintf(stderr, "Error: %s addr out of bounds\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // decimal count of bits (1 - 2000)
    int num = string_to_int(tokens[2]);
    if (num < 1 || num > GENERIC_READ_COILS_MAX ||
        (addr + num - 1 > ADDRESS_END)) {
        fprintf(stderr, "Error: %s num out of bounds\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // all checks ok; write to output struct
    request->type = READ_COILS;
    request->u.r_coils.addr = addr;
    request->u.r_coils.num = num;

    // cleanup
    free(tokens);
    free(linecopy);
    return 0;
}

static int parse_w_coil(const char *line, generic_req_t *request) {
    char *linecopy = strdup(line);
    const char *TYPE_STRING = "write_single_coil";
    const int TOKENS_EXPECTED = 3;
    int num_tokens = 0;

    char **tokens = splitline(linecopy, &num_tokens);
    if (!tokens) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(linecopy);
        return -1;
    }
    if (num_tokens != TOKENS_EXPECTED) {
        fprintf(stderr, "Error: expected %d tokens, got %d\n",
                TOKENS_EXPECTED, num_tokens);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // "write_single_coil"
    if (strcmp(tokens[0], TYPE_STRING) != 0) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // hex address (0x0000 - 0xFFFF)
    int addr = string_to_int(tokens[1]);
    if (addr < 0 || addr > GENERIC_ADDR_MAX ||
        addr < ADDRESS_START || addr > ADDRESS_END) {
        fprintf(stderr, "Error: %s addr out of bounds\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // bit value (0 or 1)
    int bit = string_to_int(tokens[2]);
    if (bit != 0 && bit != 1) {
        fprintf(stderr, "Error: %s illegal bit value\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // all checks ok; write to output struct
    request->type = WRITE_SINGLE_COIL;
    request->u.w_coil.addr = addr;
    request->u.w_coil.bit = bit;

    // cleanup
    free(tokens);
    free(linecopy);
    return 0;
}

static int parse_w_coils(const char *line, generic_req_t *request) {
    char *linecopy = strdup(line);
    const char *TYPE_STRING = "write_coils";
    const int TOKEN_MIN = 4;
    const int TOKEN_MAX = 3 + GENERIC_WRITE_COILS_MAX;
    int num_tokens = 0;

    char **tokens = splitline(linecopy, &num_tokens);
    if (!tokens) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(linecopy);
        return -1;
    }
    if (num_tokens < TOKEN_MIN || num_tokens > TOKEN_MAX) {
        fprintf(stderr, "Error: expected %d-%d tokens, got %d\n",
                TOKEN_MIN, TOKEN_MAX, num_tokens);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // "write_coils"
    if (strcmp(tokens[0], TYPE_STRING) != 0) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // hex address (0x0000 - 0xFFFF)
    int addr = string_to_int(tokens[1]);
    if (addr < 0 || addr > GENERIC_ADDR_MAX ||
        addr < ADDRESS_START || addr > ADDRESS_END) {
        fprintf(stderr, "Error: %s addr out of bounds\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // decimal count of bits (1 - 2000)
    int num = string_to_int(tokens[2]);
    if (num < 1 || num > GENERIC_WRITE_COILS_MAX ||
        addr + num - 1 > ADDRESS_END) {
        fprintf(stderr, "Error: %s illegal number of bits\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }
    if (num != num_tokens - 3) {
        fprintf(stderr, "Error: %s number of bits mismatch\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // check bits for legal values
    for (int i = 0; i < num; i++) {
        int bit = string_to_int(tokens[3+i]);
        if (bit != 0 && bit != 1) {
            fprintf(stderr, "Error: %s illegal bit value\n", TYPE_STRING);
            free(tokens);
            free(linecopy);
            return -1;
        }
    }

    // all checks ok; write to output struct
    request->type = WRITE_COILS;
    request->u.w_coils.addr = addr;
    request->u.w_coils.num = num;
    for (int i = 0; i < num; i++) {
        request->u.w_coils.bits[i] = string_to_int(tokens[3+i]);
    }

    // cleanup
    free(tokens);
    free(linecopy);
    return 0;
}

static int parse_r_registers(const char *line, generic_req_t *request) {
    char *linecopy = strdup(line);
    const char *TYPE_STRING = "read_registers";
    const int TOKENS_EXPECTED = 3;
    int num_tokens = 0;

    char **tokens = splitline(linecopy, &num_tokens);
    if (!tokens) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(linecopy);
        return -1;
    }
    if (num_tokens != TOKENS_EXPECTED) {
        fprintf(stderr, "Error: expected %d tokens, got %d\n",
                TOKENS_EXPECTED, num_tokens);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // "read_registers"
    if (strcmp(tokens[0], TYPE_STRING) != 0) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // hex address (0x0000 - 0xFFFF)
    int addr = string_to_int(tokens[1]);
    if (addr < 0 || addr > GENERIC_ADDR_MAX ||
        addr < ADDRESS_START || addr > ADDRESS_END) {
        fprintf(stderr, "Error: %s addr out of bounds\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // decimal count of registers (1 - 125)
    int num = string_to_int(tokens[2]);
    if (num < 1 || num > GENERIC_READ_REGISTERS_MAX ||
        (addr + num - 1 > ADDRESS_END)) {
        fprintf(stderr, "Error: %s num out of bounds\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // all checks ok; write to output struct
    request->type = READ_REGISTERS;
    request->u.r_regs.addr = addr;
    request->u.r_regs.num = num;

    // cleanup
    free(tokens);
    free(linecopy);
    return 0;
}

static int parse_w_register(const char *line, generic_req_t *request) {
    char *linecopy = strdup(line);
    const char *TYPE_STRING = "write_single_register";
    const int TOKENS_EXPECTED = 3;
    int num_tokens = 0;

    char **tokens = splitline(linecopy, &num_tokens);
    if (!tokens) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(linecopy);
        return -1;
    }
    if (num_tokens != TOKENS_EXPECTED) {
        fprintf(stderr, "Error: expected %d tokens, got %d\n",
                TOKENS_EXPECTED, num_tokens);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // "write_single_register"
    if (strcmp(tokens[0], TYPE_STRING) != 0) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // hex address (0x0000 - 0xFFFF)
    int addr = string_to_int(tokens[1]);
    if (addr < 0 || addr > GENERIC_ADDR_MAX ||
        addr < ADDRESS_START || addr > ADDRESS_END) {
        fprintf(stderr, "Error: %s addr out of bounds\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // 2-byte hex value (0x0000 - 0xFFFF)
    int val = string_to_int(tokens[2]);
    if (val < 0 || val > GENERIC_REGISTER_VAL_MAX) {
        fprintf(stderr, "Error: %s value out of bounds: %d\n",
                    TYPE_STRING, val);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // all checks ok; write to output struct
    request->type = WRITE_SINGLE_REGISTER;
    request->u.w_reg.addr = addr;
    request->u.w_reg.val = val;

    // cleanup
    free(tokens);
    free(linecopy);
    return 0;
}

static int parse_w_registers(const char *line, generic_req_t *request) {
    char *linecopy = strdup(line);
    const char *TYPE_STRING = "write_registers";
    const int TOKEN_MIN = 4;
    const int TOKEN_MAX = 3 + GENERIC_WRITE_REGISTERS_MAX;
    int num_tokens = 0;

    char **tokens = splitline(linecopy, &num_tokens);
    if (!tokens) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(linecopy);
        return -1;
    }
    if (num_tokens < TOKEN_MIN || num_tokens > TOKEN_MAX) {
        fprintf(stderr, "Error: expected %d-%d tokens, got %d\n",
                TOKEN_MIN, TOKEN_MAX, num_tokens);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // "write_registers"
    if (strcmp(tokens[0], TYPE_STRING) != 0) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // hex address (0x0000 - 0xFFFF)
    int addr = string_to_int(tokens[1]);
    if (addr < 0 || addr > GENERIC_ADDR_MAX ||
        addr < ADDRESS_START || addr > ADDRESS_END) {
        fprintf(stderr, "Error: %s addr out of bounds\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // decimal count of registers (1 - 123)
    int num = string_to_int(tokens[2]);
    if (num < 1 || num > GENERIC_WRITE_REGISTERS_MAX ||
        addr + num - 1 > ADDRESS_END) {
        fprintf(stderr, "Error: %s illegal number of registers\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }
    if (num != num_tokens - 3) {
        fprintf(stderr, "Error: %s number of registers mismatch\n",
                TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // check registers for legal values
    for (int i = 0; i < num; i++) {
        int val = string_to_int(tokens[3+i]);
        if (val < 0 && val > GENERIC_REGISTER_VAL_MAX) {
            fprintf(stderr, "Error: %s illegal register value: %d\n",
                    TYPE_STRING, val);
            free(tokens);
            free(linecopy);
            return -1;
        }
    }

    // all checks ok; write to output struct
    request->type = WRITE_REGISTERS;
    request->u.w_regs.addr = addr;
    request->u.w_regs.num = num;
    for (int i = 0; i < num; i++) {
        request->u.w_regs.vals[i] = string_to_int(tokens[3+i]);
    }

    // cleanup
    free(tokens);
    free(linecopy);
    return 0;
}

static int parse_wr_registers(const char *line, generic_req_t *request) {
    char *linecopy = strdup(line);
    const char *TYPE_STRING = "writeread_registers";
    const int TOKEN_MIN = 6;
    const int TOKEN_MAX = 5 + GENERIC_WRITE_REGISTERS_MAX;
    int num_tokens = 0;

    char **tokens = splitline(linecopy, &num_tokens);
    if (!tokens) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(linecopy);
        return -1;
    }
    if (num_tokens < TOKEN_MIN || num_tokens > TOKEN_MAX) {
        fprintf(stderr, "Error: expected %d-%d tokens, got %d\n",
                TOKEN_MIN, TOKEN_MAX, num_tokens);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // "writeread_registers"
    if (strcmp(tokens[0], TYPE_STRING) != 0) {
        fprintf(stderr, "Error parsing line: %s", line);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // hex address for read (0x0000 - 0xFFFF)
    int r_addr = string_to_int(tokens[1]);
    if (r_addr < 0 || r_addr > GENERIC_ADDR_MAX ||
        r_addr < ADDRESS_START || r_addr > ADDRESS_END) {
        fprintf(stderr, "Error: %s r_addr out of bounds\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // decimal count of registers to read (1 - 125)
    int r_num = string_to_int(tokens[2]);
    if (r_num < 1 || r_num > GENERIC_WRITEREAD_REGISTERS_READ_MAX ||
        r_addr + r_num - 1 > ADDRESS_END) {
        fprintf(stderr, "Error: %s illegal number of registers to read: %d\n",
                TYPE_STRING, r_num);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // hex address for write (0x0000 - 0xFFFF)
    int w_addr = string_to_int(tokens[3]);
    if (w_addr < 0 || w_addr > GENERIC_ADDR_MAX ||
        w_addr < ADDRESS_START || w_addr > ADDRESS_END) {
        fprintf(stderr, "Error: %s w_addr out of bounds\n", TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // decimal count of registers to write (1 - 121)
    int w_num = string_to_int(tokens[4]);
    if (w_num < 1 || w_num > GENERIC_WRITEREAD_REGISTERS_WRITE_MAX ||
        w_addr + w_num - 1 > ADDRESS_END) {
        fprintf(stderr, "Error: %s illegal number of registers to write:%d\n",
                TYPE_STRING, w_num);
        free(tokens);
        free(linecopy);
        return -1;
    }
    if (w_num != num_tokens - 5) {
        fprintf(stderr, "Error: %s number of registers mismatch\n",
                TYPE_STRING);
        free(tokens);
        free(linecopy);
        return -1;
    }

    // check registers for legal values
    for (int i = 0; i < w_num; i++) {
        int val = string_to_int(tokens[5+i]);
        if (val < 0 && val > GENERIC_REGISTER_VAL_MAX) {
            fprintf(stderr, "Error: %s illegal register value: %d\n",
                    TYPE_STRING, val);
            free(tokens);
            free(linecopy);
            return -1;
        }
    }

    // all checks ok; write to output struct
    request->type = WRITEREAD_REGISTERS;
    request->u.wr_regs.r_addr = r_addr;
    request->u.wr_regs.r_num = r_num;
    request->u.wr_regs.w_addr = w_addr;
    request->u.wr_regs.w_num = w_num;
    for (int i = 0; i < w_num; i++) {
        request->u.wr_regs.vals[i] = string_to_int(tokens[5+i]);
    }

    // cleanup
    free(tokens);
    free(linecopy);
    return 0;
}

static enum REQUEST_TYPE get_request_type(const char *line) {
    if (startswith(line, "read_coils ")) {
        return READ_COILS;
    } else if (startswith(line, "write_single_coil ")) {
        return WRITE_SINGLE_COIL;
    } else if (startswith(line, "write_coils ")) {
        return WRITE_COILS;
    } else if (startswith(line, "read_registers ")) {
        return READ_REGISTERS;
    } else if (startswith(line, "write_single_register ")) {
        return WRITE_SINGLE_REGISTER;
    } else if (startswith(line, "write_registers ")) {
        return WRITE_REGISTERS;
    } else if (startswith(line, "writeread_registers ")) {
        return WRITEREAD_REGISTERS;
    } else {
        return UNKNOWN_REQUEST_TYPE;
    }
}

static generic_req_t *next_request(void) {

    // In random mode, ignore stdin and just generate requests randomly
    if (randomized_mode) {
        int x = randrange(0,10);
        x++;
        return NULL;
    }

    // Otherwise, read a line from stdin and parse it into a request.
    char *line = NULL;
    size_t line_len = 0;
    ssize_t line_read = 0;
    int ret = -1; // parse return code (0 on success)
    generic_req_t request;

    line_read = getline(&line, &line_len, stdin);
    if (line_read != -1) {
        ret = parse_generic_req(line, &request);
    }
    if (line) {
        free(line);
    }

    // If parsing into request was successful, allocate on heap and return
    if (ret == 0) {
        generic_req_t *req = (generic_req_t *) malloc(sizeof(request));
        *req = request;
        return req;
    } else {
        return NULL;
    }
}

static int startswith(const char *string, const char *pattern) {
    const char *s = string;
    const char *p = pattern;
    for ( ; *p != '\0' && *s != '\0'; ++p, ++s) {
        if (*p != *s) { // character mismatch
            return 0;
        }
    }
    if (*p != '\0') { // have not reached the end of the pattern
        return 0;
    }
    return 1;
}

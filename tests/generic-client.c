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

#define LOOP             1
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
    UNKNOWN_REQUEST
};

// Prototypes for functions used only in this file
static int selftest();


void usage(const char *progname) {
    fprintf(stderr, "Usage: %s [serverIP [port]]\n", progname);
    fprintf(stderr, "Generic modbus client controlled by text requests.\n\n");

    const char *option_text =
        "  -r, --record F       record network activity in ktest file F\n"
        "  -p, --playback F     play network activity from ktest file F\n"
        "  -s, --selftest       run self-test and exit\n"
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
    int nb_loop;
    int addr;
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
    int randomized_mode = 0;
    const char *ktest_filename = NULL;
    const char *server_address = "127.0.0.1";
    int server_port = 1502;

    // Process command line arguments.
    while (1) {
        int c; // cmd line option char
        static struct option long_options[] = {
            {"record", required_argument, 0, 'r'},
            {"playback", required_argument, 0, 'p'},
            {"selftest", no_argument, 0, 's'},
            {"random", no_argument, 0, 'R'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        // getopt_long stores the option index here.
        int option_index = 0;

        c = getopt_long(argc, argv, "r:p:sRh", long_options, &option_index);

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
        case 's':
            run_self_test = 1;
            break;
        case 'R':
            randomized_mode = 1;
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

    /* Print any remaining command line arguments (not options). */
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

    /* RTU */
/*
    ctx = modbus_new_rtu("/dev/ttyUSB0", 19200, 'N', 8, 1);
    modbus_set_slave(ctx, SERVER_ID);
*/

    /* TCP */
    ctx = modbus_new_tcp(server_address, server_port);
    modbus_set_debug(ctx, TRUE);

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

    nb_loop = nb_fail = 0;
    while (nb_loop++ < LOOP) {
        for (addr = ADDRESS_START; addr < ADDRESS_END; addr++) {
            int i;

            /* Random numbers (short) */
            for (i=0; i<nb; i++) {
                tab_rq_registers[i] = (uint16_t) (65535.0*rand() / (RAND_MAX + 1.0));
                tab_rw_rq_registers[i] = ~tab_rq_registers[i];
                tab_rq_bits[i] = tab_rq_registers[i] % 2;
            }
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
    }

    /* Free the memory */
    free(tab_rq_bits);
    free(tab_rp_bits);
    free(tab_rq_registers);
    free(tab_rp_registers);
    free(tab_rw_rq_registers);

    /* Close the connection */
    modbus_close(ctx);
    modbus_free(ctx);

    return 0;
}

static int selftest() {
    int num_errors = 0;
    int num_success = 0;
    printf("Running self-tests...\n");
    printf("%d self-tests completed with %d errors.\n",
           num_success, num_errors);
    return 0;
}

#!/usr/bin/python

import random
import sys

# Random driver for the generic-client program
# Usage:
#
#  $ ./generic-random.py numpackets ADDRESS_END | \
#        ./generic-client -v [serverIP [port]] | \
#        grep '\[' > randdata.txt
#  $ cat randdata.txt | ./learn-length.py

# Request syntax:

#   read_coils <addr> <num>
#     addr    = hex address (0x0000 - 0xFFFF)
#     num     = decimal count of bits (1 - 2000)

#   write_single_coil <addr> <bit>
#     addr    = hex address (0x0000 - 0xFFFF)
#     bit     = 0 or 1

#   write_coils <addr> <num> <bit> [<bit>...]
#     addr    = hex address (0x0000 - 0xFFFF)
#     num     = decimal count of bits (1 - 1968) # max 0x07B0
#     bit     = 0 or 1

#   read_registers <addr> <num>
#     addr    = hex address (0x0000 - 0xFFFF)
#     num     = decimal count of registers (1 - 125) # max 0x07D

#   write_single_register <addr> <val>
#     addr    = hex address (0x0000 - 0xFFFF)
#     val     = 2-byte hex value (0x0000 - 0xFFFF)

#   write_registers <addr> <num> <val> [<val>...]
#     addr    = hex address (0x0000 - 0xFFFF)
#     num     = decimal count of registers (1 - 123) # max 0x07B
#     val     = 2-byte hex value (0x0000 - 0xFFFF)

#   writeread_registers <r_addr> <r_num> <w_addr> <w_num> <val> [<val>...]
#     r_addr  = hex address (0x0000 - 0xFFFF)
#     r_num   = decimal count of registers (1 - 125) # max 0x07D
#     w_addr  = hex address (0x0000 - 0xFFFF)
#     w_num   = decimal count of registers (1 - 121) # max 0x079
#     val     = 2-byte hex value (0x0000 - 0xFFFF)
#     NOTE: write occurs before read


# Constants
GENERIC_ADDR_MAX =                      0xFFFF
GENERIC_READ_COILS_MAX =                2000
GENERIC_WRITE_COILS_MAX =               0x07B0   # 1968
GENERIC_READ_REGISTERS_MAX =            0x07D    # 125
GENERIC_WRITE_REGISTERS_MAX =           0x07B    # 123
GENERIC_WRITEREAD_REGISTERS_READ_MAX =  0x07D    # 125
GENERIC_WRITEREAD_REGISTERS_WRITE_MAX = 0x079    # 121
GENERIC_REGISTER_VAL_MAX =              0xFFFF
ADDRESS_END = 99

def main(argv):
    # command line options
    N = int(argv[1]) # number of messages
    global ADDRESS_END
    ADDRESS_END = int(argv[2])

    for i in range(N):
        r = random.randint(1, 7)
        if r == 1:
            print(rand_read_coils())
        elif r == 2:
            print(rand_write_single_coil())
        elif r == 3:
            print(rand_write_coils())
        elif r == 4:
            print(rand_read_registers())
        elif r == 5:
            print(rand_write_single_register())
        elif r == 6:
            print(rand_write_registers())
        elif r == 7:
            print(rand_writeread_registers())
    return 0

def randbit():
    return random.randint(0, 1)

def randregister():
    return random.randint(0, GENERIC_REGISTER_VAL_MAX)

def rand_read_coils():
    fields = ["read_coils"]
    addr = random.randint(0, ADDRESS_END)
    num = random.randint(1, min(GENERIC_READ_COILS_MAX, ADDRESS_END-addr+1))
    fields.append(str(addr))
    fields.append(str(num))
    return " ".join(fields)

def rand_write_single_coil():
    fields = ["write_single_coil"]
    addr = random.randint(0, ADDRESS_END)
    bit = randbit()
    fields.append(str(addr))
    fields.append(str(bit))
    return " ".join(fields)

def rand_write_coils():
    fields = ["write_coils"]
    addr = random.randint(0, ADDRESS_END)
    num = random.randint(1, min(GENERIC_WRITE_COILS_MAX, ADDRESS_END-addr+1))
    fields.append(str(addr))
    fields.append(str(num))
    for i in range(num):
        fields.append(str(randbit()))
    return " ".join(fields)

def rand_read_registers():
    fields = ["read_registers"]
    addr = random.randint(0, ADDRESS_END)
    num = random.randint(1, min(GENERIC_READ_REGISTERS_MAX, ADDRESS_END-addr+1))
    fields.append(str(addr))
    fields.append(str(num))
    return " ".join(fields)

def rand_write_single_register():
    fields = ["write_single_register"]
    addr = random.randint(0, ADDRESS_END)
    val = randregister()
    fields.append(str(addr))
    fields.append(str(val))
    return " ".join(fields)

def rand_write_registers():
    fields = ["write_registers"]
    addr = random.randint(0, ADDRESS_END)
    num_upper_limit = min(GENERIC_WRITE_REGISTERS_MAX, ADDRESS_END-addr+1)
    num = random.randint(1, num_upper_limit)
    fields.append(str(addr))
    fields.append(str(num))
    for i in range(num):
        fields.append(str(randregister()))
    return " ".join(fields)

def rand_writeread_registers():
    fields = ["writeread_registers"]
    r_addr = random.randint(0, ADDRESS_END)
    r_num = random.randint(1, min(GENERIC_WRITEREAD_REGISTERS_READ_MAX,
                                  ADDRESS_END-r_addr+1))
    w_addr = random.randint(0, ADDRESS_END)
    w_num = random.randint(1, min(GENERIC_WRITEREAD_REGISTERS_WRITE_MAX,
                                  ADDRESS_END-w_addr+1))
    fields.append(str(r_addr))
    fields.append(str(r_num))
    fields.append(str(w_addr))
    fields.append(str(w_num))
    for i in range(w_num):
        fields.append(str(randregister()))
    return " ".join(fields)

if __name__ == "__main__":
    sys.exit(main(sys.argv))

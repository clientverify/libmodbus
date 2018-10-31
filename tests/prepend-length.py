#!/usr/bin/env python3

# Prepend a count of bytes to each line.
#
# That is, for the input line:
#
#   [00][01][00][00][00][08][FF][0F][00][2E][00][06][01][20]
#
# Write an output line:
#
#   14 [00][01][00][00][00][08][FF][0F][00][2E][00][06][01][20]
#
# That is all!

import re
import sys

for line in sys.stdin:
    # ignore non-alphanumeric characters like "[" and "]"
    fields = re.sub(r'\W', " ", line).split()
    print("%d %s" % (len(fields), line.strip()))

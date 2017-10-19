## Process this file with automake to produce Makefile.in.

## Copyright (C) 2017 David Pinto <david.pinto@bioch.ox.ac.uk>
##
## Copying and distribution of this file, with or without modification,
## are permitted in any medium without royalty provided the copyright
## notice and this notice are preserved.  This file is offered as-is,
## without any warranty.

bin_PROGRAMS = dsp
noinst_PROGRAMS = primeNums dsp-test

dsp_SOURCES = src/dsp.c \
  src/fpga_awg.c \
  src/rpouts.c \
  src/timer.c

primeNums_SOURCES = src/primeNums.c

dsp_test_SOURCES = src/dsp_test.c \
  src/gpioControl.c \
  src/timeControl.c

all-local:
	@echo "Done.  Now run 'make install'."

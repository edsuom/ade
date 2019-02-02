ADE EXAMPLES
======================================================


I. The Goldstein-Price test function.
-------------------------------------------------------------------------
See https://www.sfu.ca/~ssurjano/goldpr.html

In the ~/ade-examples directory, compile the executable with
$ gcc -Wall -o goldstein-price goldstein-price.c

Then run the command
$ python goldstein-price.py

This run so fast on a multi-core CPU that you can barely tell there's
anything significant going on. Because the population is fairly small
(40 individuals) in this example, there's not as much benefit from the
asynchronous multi-processing, but it still runs about twice as fast
with four CPU cores as it does with one.



II. Nonlinear curve fitting of temperature versus resistance curves of
six thermistors.
-------------------------------------------------------------------------

To see options, run
$ python thermistor.py -h

If you run the command without -h, you should see an image file
thermistor.png appear and get updated from time to time. Take a look
at it see the curve fit progressing.

On Linux, the qiv image viewer works great for this:
$ qiv -Te thermistor.png &



III. Parameter finder for AGM lead-acid battery open-circuit voltage
model
-------------------------------------------------------------------------

To see options, run
$ python voc.py -h

If you run the command without -h, you should see an image file
voc.png appear and get updated from time to time. Take a look
at it see the curve fit progressing.

On Linux, the qiv image viewer works great for this:
$ qiv -Te voc.png &

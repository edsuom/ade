/* goldstein-price.c:
   The Golstein-Price Optimization test function

   Written 2018 by Edwin A. Suominen as an example for the "ade" python
   package. This C code file is dedicated to the public domain.

   Compile with
   "gcc -Wall -o goldstein-price goldstein-price.c"

   If you don't have gcc, install it first with
   "sudo apt install gcc"

   Provide X and Y float values via STDIN, separated by a space and
   ending with a newline. Prints result float value on STDOUT ending
   with a newline.

   Enter "99999 99999" as X and Y and the overflowing result will
   terminate the program. Or just send Ctrl+C.

*/
#include <stdio.h>
#include <math.h>


float gp(float x, float y) {
  // The Goldstein-Price function
  float z, result;
  z = x + y + 1;
  z *= z;
  z *= 19 - 14*x + 3*x*x - 14*y + 6*x*y + 3*y*y;
  z += 1;
  result = z;
  z = 2*x - 3*y;
  z *= z;
  z *= 18 - 32*x + 12*x*x + 48*y - 36*x*y + 27*y*y;
  z += 30;
  result *= z;
  return result;
}


int main() {
  // Runs the Goldstein-Price function for each pair of float values
  // received via STDIN, until there is an overflowing result
  float x, y, result;

  while (1) {
    scanf("%f %f", &x, &y);
    result = gp(x, y);

    if (isinf(result))
      break;
    
    printf("%f\n", result);
  }
  return 0;
}

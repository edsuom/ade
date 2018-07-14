## ade
*Asynchronous Differential Evolution, with efficient multiprocessing*

* [API Docs](http://edsuom.com/ade/ade.html)
* [PyPI Page](https://pypi.python.org/pypi/ade/)
* [Project Page](http://edsuom.com/ade.html) at **edsuom.com**

[Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution)
(DE) is a type of genetic algorithm that works especially well for
optimizing combinations of real-valued variables. The *ade* Python
package does simple and smart population initialization, informative
progress reporting, adaptation of the vector differential scaling
factor *F* based on how much each generation is improving, and
automatic termination after a reasonable level of convergence to the
best solution.

But most significantly, *ade* performs Differential Evolution
*asynchronously*. When running on a multicore CPU or cluster, *ade*
can get the DE processing done several times faster than standard
single-threaded DE. It does this without departing in any way from the
numeric operations performed by the classic Storn and Price algorithm,
using either a randomly chosen or best candidate scheme.

You get a substantial multiprocessing speed-up *and* the
well-understood, time-tested behavior of the classic `DE/rand/1/bin`
or `DE/best/1/bin` algorithm. (You can pick which one to use.) The
very same target and base selection, mutation, scaling, and
crossover are done for a sequence of targets in the population, just
like you're used to. The underlying numeric recipe is not altered at
all, but everything runs a lot faster.

How is this possible? The answer is found in asynchronous processing
and the
[deferred lock](https://twistedmatrix.com/documents/current/api/twisted.internet.defer.DeferredLock.html)
concurrency mechanism provided by the Twisted framework. Read the detailed tutorial at http://edsuom.com/ade.html to find out more.


### License

Copyright (C) 2017-2018 by Edwin A. Suominen,
<http://edsuom.com/>:

    See edsuom.com for API documentation as well as information about
    Ed's background and other projects, software and otherwise.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the
    License. You may obtain a copy of the License at
    
      http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an "AS
    IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
    express or implied. See the License for the specific language
    governing permissions and limitations under the License.

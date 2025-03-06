Lab 9 - HPC
================

# Learning goals

In this lab, you are expected to practice the following skills:

- Evaluate whether a problem can be parallelized or not.
- Practice with the parallel package.
- Use Rscript to submit jobs.

## Problem 1

Give yourself a few minutes to think about what you learned about
parallelization. List three examples of problems that you believe may be
solved using parallel computing, and check for packages on the HPC CRAN
task view that may be related to it.

1: Cross validation in machine learning 2: caret -\> supports parallel
cross validation with doParallel 3: mlr, foreach, doParallel -? for
parallel model training. Ex) Financial risk modeling, climate
simulations, etc.

Also boot for bootstrapping and parallel for parallelize resampling.
Markov chain moten carlo rstan for stan for bayesian modeling
RcppParallel for parallel mcmc sampling nimle for customizing bayesian
inference

## Problem 2: Pre-parallelization

The following functions can be written to be more efficient without
using `parallel`:

1.  This function generates a `n x k` dataset with all its entries
    having a Poisson distribution with mean `lambda`.

``` r
fun1 <- function(n = 100, k = 4, lambda = 4) {
  x <- NULL
  
  for (i in 1:n)
    x <- rbind(x, rpois(k, lambda))
  
  return(x)
}

fun1alt <- function(n = 100, k = 4, lambda = 4) {
  # YOUR CODE HERE
  matrix(rpois(n*k, lambda = lambda), ncol = k)
  
}

# Benchmarking
microbenchmark::microbenchmark(
  fun1(100),
  fun1alt(100),
  unit = "ns"
)
```

    ## Unit: nanoseconds
    ##          expr    min     lq      mean median       uq      max neval
    ##     fun1(100) 225901 255251 387608.92 280700 296901.5 10967002   100
    ##  fun1alt(100)  12501  13101  27967.01  13601  14150.5  1378802   100

How much faster?

*Answer here.* We check the mean and we can see that it’s a lot faster.

2.  Find the column max (hint: Checkout the function `max.col()`).

``` r
# Data Generating Process (10 x 10,000 matrix)
set.seed(1234)
x <- matrix(rnorm(1e4), nrow=10)

# Find each column's max value
fun2 <- function(x) {
  apply(x, 2, max)
}

fun2alt <- function(x) {
  # YOUR CODE HERE
  x[cbind(max.col(t(x)), 1:ncol(x))]
}

# Benchmarking
bench <- microbenchmark::microbenchmark(
  fun2(x),
  fun2alt(x),
  unit = "us"
)
bench
```

    ## Unit: microseconds
    ##        expr     min       lq      mean   median       uq      max neval
    ##     fun2(x) 953.601 981.8505 1103.7800 1012.101 1059.501 3436.501   100
    ##  fun2alt(x)  77.601  88.9005  134.5011  109.751  119.701 2189.400   100

*Answer here with a plot.*

``` r
plot(bench)
```

![](lab09-hpc_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
ggplot2::autoplot(bench) +
  ggplot2::theme_minimal()
```

![](lab09-hpc_files/figure-gfm/unnamed-chunk-2-2.png)<!-- -->

## Problem 3: Parallelize everything

We will now turn our attention to non-parametric
[bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).
Among its many uses, non-parametric bootstrapping allow us to obtain
confidence intervals for parameter estimates without relying on
parametric assumptions.

The main assumption is that we can approximate many experiments by
resampling observations from our original dataset, which reflects the
population.

This function implements the non-parametric bootstrap:

``` r
my_boot <- function(dat, stat, R, ncpus = 1L) {
  
  # Getting the random indices
  n <- nrow(dat)
  idx <- matrix(sample.int(n, n*R, TRUE), nrow=n, ncol=R)
 
  # Making the cluster using `ncpus`
  # STEP 1: GOES HERE
  cl <- makePSOCKcluster(ncpus)
  # created worker nodes
  # creating cluster for parallel computing
  # STEP 2: GOES HERE
  clusterExport(cl, varlist = c("idx", "dat", "stat"), envir = environment())
  # sending the variables to all worker nodes
  # each run in isolated environment, dont have to access global variables.
  
  #change sequential apply to parallelized apply
  # STEP 3: THIS FUNCTION NEEDS TO BE REPLACED WITH parLapply
  ans <- parLapply(cl, seq_len(R), function(i) {
    stat(dat[idx[,i], , drop=FALSE])
  })
  
  # Coercing the list into a matrix
  ans <- do.call(rbind, ans)
  
  # STEP 4: GOES HERE
  stopCluster(cl)
  ans
  
}
```

1.  Use the previous pseudocode, and make it work with `parallel`. Here
    is just an example for you to try:

``` r
library(parallel)
# Bootstrap of a linear regression model
my_stat <- function(d) coef(lm(y~x, data = d))

# DATA SIM
set.seed(1)
n <- 500 
R <- 1e4
x <- cbind(rnorm(n)) 
y <- x*5 + rnorm(n)

# Check if we get something similar as lm
ans0 <- confint(lm(y~x))
cat("OLS CI \n")
```

    ## OLS CI

``` r
print(ans0)
```

    ##                  2.5 %     97.5 %
    ## (Intercept) -0.1379033 0.04797344
    ## x            4.8650100 5.04883353

``` r
ans1 <- my_boot(dat = data.frame(x,y), my_stat, R = R, ncpus = 4)
qs <- c(.025, .975)
cat("Bootstrp CI")
```

    ## Bootstrp CI

``` r
print(t(apply(ans1, 2, quantile, probs = qs)))
```

    ##                   2.5%      97.5%
    ## (Intercept) -0.1386903 0.04856752
    ## x            4.8685162 5.04351239

2.  Check whether your version actually goes faster than the
    non-parallel version:

``` r
# your code here
parallel::detectCores()
```

    ## [1] 16

``` r
#non-parallel 1 core
system.time(my_boot(dat = data.frame(x,y), my_stat, R = 4000, ncpus = 1L))
```

    ##  사용자  시스템 elapsed 
    ##    0.09    0.03    2.62

``` r
#parallel 8 core
system.time(my_boot(dat = data.frame(x,y), my_stat, R = 4000, ncpus = 8L))
```

    ##  사용자  시스템 elapsed 
    ##    0.21    0.12    1.11

It does go faster. Non parallel takes 5.19 while parallel takes 3.72.

## Problem 4: Compile this markdown document using Rscript

Once you have saved this Rmd file, try running the following command in
your terminal:

``` bash
Rscript --vanilla -e 'rmarkdown::render("[full-path-to-your-Rmd-file.Rmd]")' &
```

Where `[full-path-to-your-Rmd-file.Rmd]` should be replace with the full
path to your Rmd file… :).

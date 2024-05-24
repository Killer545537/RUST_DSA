///T.C. = O(√n)
pub fn divisors(num: i32) -> Vec<i32> { //This way we can also check if a number is prime
    let mut divisors = Vec::new();
    let mut i = 1;

    while i * i <= num {
        if num % i == 0 {
            divisors.push(i);
            if num / i != i { //To avoid the same factor appearing twice if the number is a perfect square
                divisors.push(num / i);
            }
        }
        i += 1;
    }

    divisors.sort();
    divisors
}

pub fn gcd(a: i32, b: i32) -> i32 {
    //We use the Euclidean Algorithm to find the gcd(a,b) = gcd(a % b, b) (a > b)
    if a == 0 {
        return b;
    } else if b == 0 {
        return a;
    }

    if a > b {
        return gcd(a % b, b);
    } else {
        return gcd(b % a, a);
    }
}

pub fn prime_factors(mut n: i32) -> Vec<i32> {
    let mut ans = Vec::new();

    let mut i = 2;
    while i * i <= n {
        if n % i == 0 {
            ans.push(i);

            while n % i == 0 {
                n /= i;
            }
        }

        i += 1;
    }

    if n != 1 {
        ans.push(n);
    }

    ans
}

pub fn sieve_of_erathosthenes(n: usize) -> Vec<bool>{
    let mut sieve = vec![true; n + 1];
    sieve[0]= false;
    if n >= 1 {
        sieve[1] = false;
    }

    for num in 2..n {
        if sieve[num] {
            let mut multiple = num * num;
            while multiple <= n {
                sieve[multiple] = false; //Mark all the multiples of the prime as composite
                //sieve[multiple] = num to store the minimum prime factor
                multiple += num;
            }

            // (num*num ..=n).step_by(num).for_each(|i| sieve[i] = false)
        }
    }

    sieve
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divisor_test() {
        assert_eq!(divisors(1), vec![1]);
        assert_eq!(divisors(7), vec![1, 7]);
        assert_eq!(divisors(49), vec![1, 7, 49]);
    }

    #[test]
    fn gcd_test() {
        assert_eq!(gcd(60, 90), 30);
        assert_eq!(gcd(1, 2), 1);
        assert_eq!(gcd(0, 4), 4);
    }

    #[test]
    fn prime_factors_test() {
        assert_eq!(prime_factors(3), vec![3]);
        assert_eq!(prime_factors(780), vec![2, 3, 5, 13])
    }
}
/*The binary system is base 2, i.e. 0 and 1. The binary of a number can be printed by "{:b}", x.
The 1s complement of a number is formed by negating each bit
The 2s complement is 1 + 1s complement
These are ways to represent integers(especially for negatives since positives are obvious)
Most languages use 2s complement to represent negative numbers
AND & is 1 if all are 1
OR | is 1 if any is 1
XOR ^ is 1 if odd number of 1s are present
NOT ! converts 1->0 and 0->1
There are two shifts, left shift << (1101 -> 11010) (equivalent to *2) and right shift >> (1101 -> 110) (equivalent to /2) */

///Swapping two numbers without the need of a second variable
pub fn swap(a: &mut i32, b: &mut i32) { //std::mem::swap(&mut a, &mut b) does the same thing
    println!("{a} and {b}");

    *a = *a ^ *b; // a = a ^ b
    *b = *a ^ *b; // b = (a ^ b) ^ b = a
    *a = *a ^ *b; // a = (a ^ b) ^ a = a

    println!("{a} and {b}");
}

///Check if the kth but is set (from the right and using 0 based indexing)
pub fn check_kth_bit(num: i32, k: usize) -> bool {
    //1 & 1 = 1 and 0 & 1 = 0
    let mask = 1 << k; // 0..0 1 0..0

    return if num & mask == 0 {
        false
    } else {
        true
    };
}

///Set the kth bit from the right end if not set
pub fn set_kth_bit(num: &mut i32, k: usize) {
    // 0 | 1 = 1 and 1 | 1 = 1
    let mask = 1 << k;

    *num = *num | mask;
}

pub fn clear_kth_bit(num: &mut i32, k: usize) {
    let mask = !(1 << k);

    *num = *num & mask;
}

pub fn toggle_kth_bit(num: &mut i32, k: usize) {
    // 1 ^ 1 = 0 and 0 ^ 1 = 1
    let mask = 1 << k;

    *num = *num ^ mask;
}

///Unset the rightmost set bit
pub fn remove_last_set_bit(num: &mut i32) {
    *num = *num & (*num - 1);
}

pub fn is_power_of_2(num: i32) -> bool {
    //If the number, on removing the rightmost set bit is 0
    return if num & (num - 1) == 0 {
        true
    } else {
        false
    };
}

pub fn count_set_bits(mut num: i32) -> i32 {
    let mut count = 0;

    while num > 1 {
        count += num & 1; //Odd check
        num >>= 1;
    }
    if num == 1 {
        count += 1;
    }

    count
    //OR we can turn off the rightmost set bit till num != 0
}

pub fn count_set_bits_better(mut num: i32) -> i32 { //num.count_ones does the same thing
    let mut count = 0;

    while num != 0 {
        num &= num - 1;
        count += 1;
    }

    count
}

pub fn min_bit_flips(start: i32, goal: i32) -> i32 {
    //start ^ goal will have the same number of set bits as the number of bit flips required
    count_set_bits_better(start ^ goal)
}

pub fn subsets(arr: &[i32]) -> Vec<Vec<i32>> {
    let mut ans: Vec<Vec<i32>> = Vec::with_capacity(2usize.pow(arr.len() as u32));
    let subsets = 1 << arr.len();

    for builder in 0..subsets {
        let mut set = Vec::new();
        for i in 0..arr.len() {
            if builder & (1 << i) != 0 { //We need != 0 since == 1 won't work (no truthy values in Rust)
                set.push(arr[i]);
            }
        }
        ans.push(set);
    }

    ans
}

///Find the number which appears once where all others appear twice
pub fn single_number(arr: &[i32]) -> i32 {
    // x ^ x = 0, and a ^ 0 = a, so all duplicated XOR to 0
    arr.iter().fold(0, |acc, &num| acc ^ num)
}

///Find the number which appears once where all others appear thrice
pub fn single_number_2(arr: &[i32]) -> i32 {
    //Very unintuitive, need to read again
    let (mut ones, mut twos) = (0, 0);

    for &i in arr {
        ones = (ones ^ i) & !twos;
        twos = (twos ^ i) & !ones;
    }

    ones
}

//Find the two numbers which appear once where all others appear twice
pub fn single_number_3(arr: &[i32]) -> (i32, i32) {
    let xor = arr.iter().fold(0, |x, &num| x ^ num);
    //Now, xor is 1 when the bits of a and b (answers) do not match
    //Also, the other numbers are in pairs, so x1 == x2 have the ith bit matching
    //Thus, we can separate arr into -> ith bit set | ith bit not set, then find the single number in each of these buckets
    let rightmost_set_bit = xor & -xor;
    let mut x = 0;
    let mut y = 0;

    for &num in arr {
        if num & rightmost_set_bit == 0 {
            x ^= num;
        } else {
            y ^= num;
        }
    }

    (x, y)
}

pub fn xor_till_n(n: i32) -> i32 {
    //1 ^ 2 ^ 3 ^ .. (4k+3) = 0
    //f(4k+3) = 0, f(n = 4k) = n, f(n = 4k + 1) = 1, f(n = 4k + 2) = n + 1
    match n % 4 {
        0 => n,
        1 => 1,
        2 => n + 1,
        3 => 0,
        _ => unreachable!()
    }
}

///Divide without using division operator
pub fn divide(dividend: i32, divisor: i32) -> i32 {
    //To find x/y, we use the fact that x = y * (∑2^i), thus we find these i's and thus add 2*i to find the quotient
    if dividend == divisor {
        return 1;
    }
    //true is +ve
    let mut sign = true;

    if dividend >= 0 && divisor < 0 {
        sign = false;
    } else if dividend <= 0 && divisor > 0 {
        sign = false;
    }

    let mut dividend = (dividend as i64).abs();
    let divisor = (divisor as i64).abs();
    let mut quotient = 0;

    while dividend >= divisor {
        let mut count = 0;

        while dividend >= divisor << (count + 1) {
            count += 1;
        }

        quotient += 1 << count;
        dividend -= divisor << count;
    }

    if quotient == 1 << 31 && sign {
        return i32::MAX;
    }
    if quotient == 1 << 31 && !sign {
        return i32::MIN;
    }

    return if sign {
        quotient
    } else {
        -quotient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_test() {
        let mut a = 3;
        let mut b = 4;
        swap(&mut a, &mut b);
        assert_eq!(a, 4);
        assert_eq!(b, 3);
    }

    #[test]
    fn check_kth_bit_test() {
        assert_eq!(check_kth_bit(0b10101, 0), true);
        assert_eq!(check_kth_bit(0b10101, 1), false);
    }

    #[test]
    fn set_kth_bit_test() {
        let mut x = 0b1001;
        set_kth_bit(&mut x, 0);
        assert_eq!(x, 0b1001);
        set_kth_bit(&mut x, 1);
        assert_eq!(x, 0b1011);
    }

    #[test]
    fn clear_kth_bit_test() {
        let mut x = 0b1101;
        clear_kth_bit(&mut x, 2);
        assert_eq!(x, 0b1001);
        clear_kth_bit(&mut x, 1);
        assert_eq!(x, 0b1001);
    }

    #[test]
    fn toggle_kth_bit_test() {
        let mut x = 0b1101;
        toggle_kth_bit(&mut x, 2);
        assert_eq!(x, 0b1001);
        toggle_kth_bit(&mut x, 2);
        assert_eq!(x, 0b1101);
    }

    #[test]
    fn remove_last_set_bit_test() {
        let mut x = 0b1010100;
        remove_last_set_bit(&mut x);
        assert_eq!(x, 0b1010000);

        let mut x = 0b100;
        remove_last_set_bit(&mut x);
        assert_eq!(x, 0);
    }

    #[test]
    fn power_2_test() {
        assert_eq!(is_power_of_2(16), true);
        assert_eq!(is_power_of_2(1), true);
    }

    #[test]
    fn number_of_set_bits_test() {
        assert_eq!(count_set_bits(4), 1);
        assert_eq!(count_set_bits(0b1011), 3);
        assert_eq!(count_set_bits(0), 0);
    }

    #[test]
    fn number_of_set_bits_test_better() {
        assert_eq!(count_set_bits_better(4), 1);
        assert_eq!(count_set_bits_better(0b1011), 3);
        assert_eq!(count_set_bits_better(0), 0);
    }

    #[test]
    fn min_flips_test() {
        assert_eq!(min_bit_flips(10, 7), 3);
        assert_eq!(min_bit_flips(3, 4), 3);
    }

    #[test]
    fn subset_test() {
        assert_eq!(subsets(&[1, 2, 3]), vec![
            vec![],
            vec![1],
            vec![2],
            vec![1, 2],
            vec![3],
            vec![1, 3],
            vec![2, 3],
            vec![1, 2, 3],
        ]);
    }

    #[test]
    fn single_test() {
        assert_eq!(single_number(&[2, 2, 1]), 1);
        assert_eq!(single_number(&[4, 1, 2, 1, 2]), 4)
    }

    #[test]
    fn single_again_test() {
        assert_eq!(single_number_2(&[2, 2, 3, 2]), 3);
        assert_eq!(single_number_2(&[0, 1, 0, 1, 0, 1, 99]), 99);
    }

    #[test]
    fn single_still_test() {
        assert!(single_number_3(&[1, 2, 1, 3, 2, 5]) == (3, 5) || single_number_3(&[1, 2, 1, 3, 2, 5]) == (5, 3));
        assert!(single_number_3(&[-1, 0]) == (-1, 0) || single_number_3(&[-1, 0]) == (0, -1));
    }
}
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
    let subsets  = 1 << arr.len();

    for builder in 0.. subsets {
       let mut set = Vec::new();
        for i in 0.. arr.len() {
            if builder & (1 << i) != 0 { //We need != 0 since == 1 won't work (no truthy values in Rust)
                set.push(arr[i]);
            }
        }
        ans.push(set);
    }

    ans
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
        assert_eq!(subsets(&[1,2,3]), vec![
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
}
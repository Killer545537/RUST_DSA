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
    }
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
}
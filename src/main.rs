#![allow(dead_code)]


mod arrays;
mod binary_search;

fn find_floor(arr: &[i32], target: i32) -> Option<usize> {
    let (mut left, mut right)  = (0, arr.len());

    while left < right {
        let mid = (left + right) / 2;

        if arr[mid] > target {
            right = mid;
        } else if arr[mid] == target {
            return Some(mid);
        } else {
            left = mid + 1;
        }
    }

    None
}

fn main() {
    let arr = [1,3,4,4,5,10,32];
    let first = arr.partition_point(|&x| x <= 4);

    println!("{first}")
}
#![allow(dead_code)]

mod arrays;
mod binary_search;
mod linked_lists;

fn main() {
    let arr = [10, 20, 30, 40];

    let v: Option<i32> = arr.iter().enumerate().map(|(i, _)| std::cmp::max(arr[..=i].iter().sum::<i32>(), arr[i+1..].iter().sum::<i32>())).min();
    println!("{v:?}")
}
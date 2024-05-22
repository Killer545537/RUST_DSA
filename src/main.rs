#![allow(dead_code)]


mod arrays;
mod bit_manipulation;
mod binary_search;
mod linked_lists;
mod doubly_linked_list;
mod recursion;
mod greedy;

fn main() {
    let x: usize = 2;
    let y = 1 << x;
    println!("{y}")
}

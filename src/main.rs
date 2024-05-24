#![allow(dead_code)]



mod arrays;
mod bit_manipulation;
mod binary_search;
mod linked_lists;
mod doubly_linked_list;
mod recursion;
mod greedy;
mod math;


fn main() {
    let v = vec![12,3,4,12];

    let y = v.iter().filter(|&&x| x % 2 == 1).count();
}

#![allow(dead_code)]


use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use binary_tress::*;

mod arrays;
mod bit_manipulation;
mod binary_search;
mod linked_lists;
mod doubly_linked_list;
mod recursion;
mod greedy;
mod math;
mod dynamic_programming;
mod binary_tress;
mod two_pointer;

fn main() {
    let mut map = HashMap::new();
    map.insert(4,3);
    map.insert(10,3);

    let x = map[&4];

    println!("{x}");
}
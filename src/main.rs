#![allow(dead_code)]


use std::cell::RefCell;
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

fn main() {
    let mut root = Node::new(10);
    root.add_left(5);
    root.add_right(3);

    println!("{root:?}");
    children_sum_property(Some(Rc::new(RefCell::new(root.clone()))));
    println!("{root:?}")
}
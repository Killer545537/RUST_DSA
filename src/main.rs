#![feature(linked_list_retain)]
#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

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
mod graphs;
mod stack_queues;
mod expression_conversion;

fn c<F: FnOnce() + 'static>(f: F) {
    f()
}

fn main() {
    let v = Rc::new(RefCell::new(vec![1,2,3]));
    v.borrow_mut().push(4);

    let v_clone = v.clone();
    c(move || {
       println!("{:?}", v_clone.borrow());
        v_clone.borrow_mut().push(5);
    });

    println!("{:?}", v);
}
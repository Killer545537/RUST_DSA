#![feature(linked_list_retain)]
#![allow(dead_code)]

use crate::dynamic_programming::{matrix_chain_multiplication, optimal_parenthesis};

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


fn main() {
    let matrices = vec![3, 2, 4, 2, 5];
    let (operations, split) = matrix_chain_multiplication(matrices);
    let x = optimal_parenthesis(&split, 1, split.len() - 1);

    println!("{}", x);
}
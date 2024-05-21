#![allow(dead_code)]


use std::collections::HashMap;
use std::fmt::{Display, Formatter};

mod arrays;
mod binary_search;
mod linked_lists;
mod doubly_linked_list;
mod recursion;

enum Color {
    Red,
    Green,
    Blue,
}

impl Display for Color {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Color::Red => write!(f, "🟥"),
            Color::Green => write!(f, "🟩"),
            Color::Blue => write!(f, "🟦")
        }
    }
}

fn idk1() {
    const SIZE: usize = 4;
    fn idk2() {
        println!("{}", SIZE);
    }
}

fn main() {
}
#![allow(dead_code)]


mod arrays;
mod binary_search;
mod linked_lists;
mod doubly_linked_list;
mod recursion;

fn is_palindrome(s: &str) -> bool {
    s.chars().eq(s.clone().chars().rev())
}
fn main() {
    let s = "a";
    println!("{}", is_palindrome(s));
}
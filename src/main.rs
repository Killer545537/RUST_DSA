#![allow(dead_code)]

mod arrays;
mod binary_search;
mod linked_lists;
mod doubly_linked_list;

fn main() {
    let mut ll = linked_lists::LinkedList::new();
    ll.push_front(2);
    ll.push_back(4);
    ll.push_front(1);
    println!("{}", ll);
}
/*  A linked list is a collection of data elements (node) where the linear order is given by pointers. It has a dynamic size.
It has T.C. = O(1) for insertion/deletion at beginning, O(n) for insertion/deletion at the end, insertion at a position and searching
It can be used to implement stacks, queues, etc.
Rust has a built-in linked-list data-type LinkedList in the std::collections module (This is actually a doubly linked list)*/

use std::fmt::{Display, Formatter};
use std::ptr::write;

struct Node<T> {
    value: T,
    next: Option<Box<Node<T>>>,
    //Option is used since the next may point to a NULL
    //Box is used to allocate memory on the heap, since Rust requires the size of a struct to be known at compile time, it disallows the use of recursive types
}

impl<T> Node<T> {
    fn new(value: T) -> Self {
        Node {
            value,
            next: None,
        }
    }
}

pub struct LinkedList<T> {
    head: Option<Box<Node<T>>>,
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        LinkedList { head: None }
    }

    pub fn push_front(&mut self, value: T) {
        let mut node = Box::new(Node::new(value));

        match self.head.take() {
            None => self.head = Some(node),
            Some(old_head) => {
                node.next = Some(old_head);
                self.head = Some(node);
            }
        }
    }

    pub fn push_back(&mut self, value: T) {
        let node = Box::new(Node::new(value));
        let mut current = &mut self.head;

        while let Some(ref mut node) = current {
            current = &mut node.next;
        }

        *current = Some(node);
    }


}

impl<T: Display> Display for LinkedList<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
       let mut current = &self.head;
        let mut result = String::new();

        while let Some(node) = current {
            result.push_str(&format!("{}", node.value));
            if node.next.is_some() {
                result.push_str(", ");
            }
            current = &node.next;
        }

        write!(f, "[{}]", result)
    }
}
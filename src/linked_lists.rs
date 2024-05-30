/*  A linked list is a collection of data elements (node) where the linear order is given by pointers. It has a dynamic size. The memory locations are not contiguous.
It has T.C. = O(1) for insertion/deletion at beginning, O(n) for insertion/deletion at the end, insertion at a position and searching
It can be used to implement stacks, queues, etc. It is used in browser history, sparse data structures (where most of the data is 0/empty), playlists, etc.
Rust has a built-in linked-list data-type LinkedList in the std::collections module (This is actually a doubly linked list)*/

use std::fmt::{Display, Formatter};

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

impl<T: Copy> LinkedList<T> {
    pub fn new() -> Self {
        LinkedList { head: None }
    }
    ///Add an element to the front. T.C. = O(1)
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
    ///Add an element to the end of the list. T.C. = O(n)
    pub fn push_back(&mut self, value: T) {
        let node = Box::new(Node::new(value));
        let mut current = &mut self.head;

        while let Some(ref mut node) = current {
            current = &mut node.next;
        }

        *current = Some(node);
    }
    //option.take() is a method which takes the value out of the option, leaving a None in its place.
    pub fn reverse(&mut self) {
        let mut current_head = self.head.take();

        while let Some(mut current_node) = current_head {
            let current_next = current_node.next.take();
            current_node.next = self.head.take();
            self.head = Some(current_node);
            current_head = current_next;
        }
    }
    ///Delete the first node
    pub fn delete_front(&mut self) -> Result<T, &'static str> {
        match self.head.take() { //Take head since we are 'defo' changing it
            None => Err("List is empty"),//If the head is none, then the list is empty and cannot be deleted
            Some(mut old_head) => {
                //Free is done automatically by .take()
                self.head = old_head.next.take();
                Ok(old_head.value)
            }
        }
    }
    ///Delete the last node in the list
    pub fn delete_back(&mut self) -> Result<T, &'static str> {
        match self.head.as_mut() { //Take head as mut since we are modifying it 'maybe'
            None => Err("List is empty"),
            Some(head) => {
                if head.next.is_none() {
                    return Ok(self.head.take().unwrap().value);
                }
                let mut current = head;
                while current.next.as_ref().unwrap().next.is_some() { //Move to the second last node
                    current = current.next.as_mut().unwrap();
                }
                let result = current.next.take().unwrap().value;
                Ok(result)
            }
        }
    }
}

impl<T: Copy + PartialEq> LinkedList<T> {
    ///Finding an element in the list. T.C. = O(n)
    pub fn search(&self, value: T) -> bool {
        let mut current = &self.head;

        while let Some(node) = current {
            if node.value == value {
                return true;
            }
            current = &node.next;
        }

        false //value not in the list
    }
}

//Println
impl<T: Display> Display for LinkedList<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut current = &self.head;
        let mut result = String::new();

        while let Some(node) = current {
            result.push_str(&format!("{}", node.value));
            if node.next.is_some() {
                result.push_str(" ➡️ ");
            }
            current = &node.next;
        }
        result.push_str(" ➡️ NULL");
        write!(f, "[{}]", result)
    }
}

//The following implementation is for LeetCode Questions
#[derive(PartialEq, Eq, Clone, Debug)]
struct ListNode {
    val: i32,
    next: Option<Box<ListNode>>,
}

impl ListNode {
    fn new(val: i32) -> Self {
        ListNode {
            val,
            next: None,
        }
    }
}

///Given two numbers in a linked list in reverse order (123 => 3->2->1->None)
pub fn add_two_numbers(l1: Option<Box<ListNode>>, l2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    fn helper(l1: Option<Box<ListNode>>, l2: Option<Box<ListNode>>, carry: i32) -> Option<Box<ListNode>> {
        if l1.is_none() && l2.is_none() && carry == 0 {
            return None;
        }

        let l1 = l1.unwrap_or(Box::new(ListNode::new(0)));
        let l2 = l2.unwrap_or(Box::new(ListNode::new(0)));

        let sum = l1.val + l2.val + carry;
        let mut node = ListNode::new(sum % 10);
        node.next = helper(l1.next, l2.next, sum / 10);

        Some(Box::new(node))
    }

    helper(l1, l2, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
}
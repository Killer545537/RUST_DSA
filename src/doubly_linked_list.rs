use std::cell::RefCell;
use std::rc::{Rc, Weak};

struct Node<T> {
    value: T,
    next: Option<Rc<RefCell<Node<T>>>>,
    prev: Option<Weak<RefCell<Node<T>>>>,
    /* RefCell is used to enforce borrowing rules at runtime instead of compile time
     RefCell allows us to have mutable references even when there are immutable references to it
     Rc (Reference Counting) allows data to have multiple owners, the object is dropped when there are no more references
     Weak is a non-owning Rc. Weak can be upgraded to Rc by .upgrade() (-> Option if the value is still present)
     Weak does not increase the reference count and the value is dropped when there are only Weak pointers to it
     Weak is used to avoid circular referencing  */
}

impl<T> Node<T> {
    fn new(value: T) -> Self {
        Node {
            value,
            next: None,
            prev: None,
        }
    }
}

impl<T> From<Node<T>> for Option<Rc<RefCell<Node<T>>>> {
    //This is used to convert a Node to a NodePtr
    fn from(value: Node<T>) -> Self { Some(Rc::new(RefCell::new(Node::new(value)))) }
}

type NodePtr<T> = Rc<RefCell<Node<T>>>; //This is a type-def

pub struct DoublyLinkedList<T> {
    head: Option<NodePtr<T>>,
    tail: Option<NodePtr<T>>,
}

impl<T: Copy> DoublyLinkedList<T> {
    pub fn new() -> Self {
        DoublyLinkedList {
            head: None,
            tail: None,
        }
    }
    ///Add an element to the front
    pub fn push_front(&mut self, value: T) {
        //Create the new node
        let mut node = Node::new(value);

        match &mut self.head.take() {
            None => { //If the list is empty
                self.head = node.into(); //Make the new node as the head
                self.tail = self.head.clone(); //Make the tail as the new node as well
            }
            Some(old_head) => {
                node.next = Some(old_head.clone()); //new_node -> next = old_head
                self.head = node.into(); //Make the new node the head
                if let Some(h) = &self.head { //This head is the new node
                    //Make the prev of the old_head the new node
                    old_head.borrow_mut().prev = Some(Rc::downgrade(&h));
                }
            }
        }
    }
    ///Add an element to the end
    pub fn push_back(&mut self, value: T) {
        let mut node = Node::new(value);

        match &mut self.tail.take() {
            None => { //If the list is empty, same thing as before
                self.head = node.into(); //Make the new node as the head
                self.tail = self.head.clone(); //Make the tail as the new node as well
            }
            Some(old_tail) => {
                node.prev = Some(Rc::downgrade(&old_tail));
                self.tail = node.into();
                old_tail.borrow_mut().next = self.tail.clone();
            }
        }
    }
    ///Delete an element from the end
    pub fn pop_back(&mut self) -> Result<T, &'static str> {
        match self.tail.take() {
            None => Err("List is empty"), //If the list is empty
            Some(tail) => {
                let mut tail = tail.borrow_mut();
                let prev = tail.prev.take();

                match prev {
                    None => {
                        self.head.take();
                    }
                    Some(prev) => {
                        let prev = prev.upgrade();
                        if let Some(prev) = prev {
                            prev.borrow_mut().next = None;
                            self.tail = Some(prev);
                        }
                    }
                };

                Ok(tail.value)
            }
        }
    }
    ///Delete an element from the beginning
    pub fn pop_front(&mut self) -> Result<T, &'static str> {
        match self.head.take() {
            None => Err("List is empty"),
            Some(head) => {
                let mut head = head.borrow_mut();
                let next = head.next.take();

                match next {
                    None => {
                        self.head.take();
                    }
                    Some(next) => {
                        next.borrow_mut().prev = None;
                        self.head = Some(next);
                    }
                };

                Ok(head.value)
            }
        }
    }
}
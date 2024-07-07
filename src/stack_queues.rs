/*
A stack is a list of elements in which elements can only be inserted or deleted from the top of the stack. It is a Last-In-First-Out data structure. Push is used for insertion and pop for deletion.
It is used in recursion (recursion stack), expression evaluation (infix, prefix, postfix), tree traversal, browser history, etc.
A queue is a list of elements in which elements can be inserted from the rear and deleted from the front. It is a First-In-First-Out data structure. Push/enqueue is used for insertion and pop/dequeue for deletion.
It is used in sharing resources, CPU scheduling, call center, etc.
 */
use std::cmp::Ordering;
use std::collections::{HashMap, LinkedList, VecDeque};
use std::fmt::{Display, Formatter};
use std::mem::take;

struct Stack<T: Copy, const SIZE: usize> {
    array: [Option<T>; SIZE],
    top: Option<usize>,
}

#[derive(Debug, PartialEq)]
enum StackError {
    Overflow,
    Underflow,
}

impl Display for StackError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StackError::Overflow => write!(f, "Stack Overflow: Unable to push items as the stack is full"),
            StackError::Underflow => write!(f, "Stack Underflow: Unable to pop items as the stack is empty"),
        }
    }
}

impl<T: Copy, const SIZE: usize> Stack<T, SIZE> {
    pub fn new() -> Self {
        Stack {
            array: [None; SIZE],
            top: None,
        }
    }

    pub fn push(&mut self, item: T) -> Result<(), StackError> {
        match self.top {
            Some(top) if top >= SIZE - 1 => Err(StackError::Overflow),
            Some(top) => {
                self.top = Some(top + 1);
                self.array[top + 1] = Some(item);
                Ok(())
            }
            None => {
                self.top = Some(0);
                self.array[0] = Some(item);
                Ok(())
            }
        }
    }

    pub fn pop(&mut self) -> Result<T, StackError> {
        match self.top {
            None => Err(StackError::Underflow),
            Some(top) => {
                self.top = if top == 0 {
                    None
                } else {
                    Some(top - 1)
                };
                Ok(self.array[top].take().unwrap())
            }
        }
    }

    pub fn peek(&self) -> Option<&T> {
        match self.top {
            None => None,
            Some(top) => self.array[top].as_ref()
        }
    }
}

struct Queue<T: Copy, const SIZE: usize> {
    array: [Option<T>; SIZE],
    front: Option<usize>,
    rear: Option<usize>,
}

#[derive(Debug, PartialEq)]
enum QueueError {
    Overflow,
    Underflow,
}

impl Display for QueueError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            QueueError::Overflow => write!(f, "Queue Overflow: Unable to push items as the queue is full"),
            QueueError::Underflow => write!(f, "Queue Underflow: Unable to pop items as the queue is empty")
        }
    }
}

impl<T: Copy, const SIZE: usize> Queue<T, SIZE> {
    fn new() -> Self {
        Queue {
            array: [None; SIZE],
            front: None,
            rear: None,
        }
    }

    fn enqueue(&mut self, item: T) -> Result<(), QueueError> {
        match self.rear {
            Some(rear) if rear >= SIZE => Err(QueueError::Overflow),
            Some(rear) => {
                self.array[rear] = Some(item);
                self.rear = Some(rear + 1);
                Ok(())
            }
            None => {
                self.array[0] = Some(item);
                self.front = Some(0);
                self.rear = Some(1);
                Ok(())
            }
        }
    }

    fn dequeue(&mut self) -> Result<T, QueueError> {
        match self.front {
            None => Err(QueueError::Underflow),
            Some(front) if front == self.rear.unwrap() => {
                self.front = None;
                self.rear = None;
                Err(QueueError::Underflow)
            }
            Some(front) => {
                let item = self.array[front].take();
                self.front = Some(front + 1);

                if self.front == self.rear {
                    self.front = None;
                    self.rear = None;
                }

                Ok(item.unwrap())
            }
        }
    }
}

/*
Instead of these definitions of Stack and Queue, we will use VecDeque(stands for Doubly Ended Vector)
It can work like a stack and queue
To use it like a stack use push_back and pop_back
To use it like a queue use push_back and pop_front
 */

///Implement Stack using a Queue
struct MyStack {
    queue: VecDeque<i32>,
}

impl MyStack {
    fn new() -> Self {
        MyStack {
            queue: VecDeque::new()
        }
    }

    fn push(&mut self, x: i32) {
        let size = self.queue.len();
        self.queue.push_back(x);

        for _ in 0..size {
            let y = self.queue.pop_front().unwrap();
            self.queue.push_back(y);
        }
    }

    fn pop(&mut self) -> i32 {
        self.queue.pop_front().unwrap() //It is guaranteed that all pop calls are valid
    }

    fn top(&self) -> i32 {
        *self.queue.front().unwrap()
    }

    fn empty(&self) -> bool {
        self.queue.is_empty()
    }
}

///Implement Queue using stack
struct MyQueue {
    stack_input: VecDeque<i32>,
    stack_output: VecDeque<i32>,
}

impl MyQueue {
    fn new() -> Self {
        MyQueue {
            stack_input: VecDeque::new(),
            stack_output: VecDeque::new(),
        }
    }

    fn push(&mut self, x: i32) {
        self.stack_input.push_back(x);
    }

    fn pop(&mut self) -> i32 {
        return if !self.stack_output.is_empty() {
            self.stack_output.pop_back().unwrap()
        } else {
            while !self.stack_input.is_empty() {
                self.stack_output.push_back(self.stack_input.pop_back().unwrap());
            }
            self.stack_output.pop_back().unwrap()
        };
    }

    fn peek(&mut self) -> i32 {
        return if !self.stack_output.is_empty() {
            *self.stack_output.back().unwrap()
        } else {
            while !self.stack_input.is_empty() {
                self.stack_output.push_back(self.stack_input.pop_back().unwrap());
            }
            *self.stack_output.back().unwrap()
        };
    }

    fn empty(&self) -> bool {
        self.stack_input.is_empty() && self.stack_output.is_empty()
    }
}

pub fn is_valid(s: String) -> bool {
    let mut stack = VecDeque::new();
    let brackets: HashMap<char, char> = [(')', '('), (']', '['), ('}', '{')].iter().cloned().collect();

    for bracket in s.chars() {
        match brackets.get(&bracket) {
            None => stack.push_back(bracket),
            Some(&matching_bracket) => {
                if stack.pop_back() != Some(matching_bracket) {
                    return false;
                }
            }
        }
    }

    stack.is_empty()
}

pub fn next_greater_elements(nums: Vec<i32>) -> Vec<i32> {
    let n = nums.len();
    let mut next_greater_element = vec![-1; n];
    let mut stack = VecDeque::new();

    //Duplicate the array at the end (index % length will do it)
    for i in (0..=2 * n - 1).rev() { //Start from the right end
        //if the top element is less,
        while !stack.is_empty() && *stack.back().unwrap() <= nums[i % n] {
            stack.pop_back();
        }

        if i < n { //Calculate only for the left half
            if !stack.is_empty() {
                next_greater_element[i] = *stack.back().unwrap();
            } else {
                next_greater_element[i] = -1;
            }
        }

        stack.push_back(nums[i % n]);
    }

    next_greater_element
}

//An LRU (Least Recently Used) Cache is one of the most common caching systems. It removes the least recently used item. It requires delete and insert operations to be O(1) and fast lookups (HashMap)

#[derive(Default, Clone)]
struct Node {
    key: i32,
    value: i32,
}

//This uses an unstable function retain
struct LRUCache {
    //A Box is needed so that Node is owned by map and list
    map: HashMap<i32, Box<Node>>,
    list: LinkedList<Box<Node>>, //This is a Doubly Linked List
    capacity: usize,
}

impl LRUCache {
    fn new(capacity: usize) -> Self {
        LRUCache {
            map: HashMap::new(),
            list: LinkedList::new(),
            capacity,
        }
    }

    fn get(&mut self, key: i32) -> Option<i32> {
        match self.map.get_mut(&key) {
            None => None,
            Some(node) => {
                let node = take(node);
                let node_clone = node.clone();
                self.list.retain(|n| n.key != key);
                self.list.push_front(node);
                Some(node_clone.value)
            }
        }
    }

    fn put(&mut self, key: i32, value: i32) {
        if let Some(node) = self.map.get_mut(&key) {
            node.value = value;
            self.list.retain(|n| n.key != key);

            let node = node.clone();
            self.list.push_front(node);
        } else {
            if self.map.len() == self.capacity {
                let last_node = self.list.pop_back().unwrap();
                self.map.remove(&last_node.key);
            }

            let node = Box::new(Node { key, value });
            self.list.push_front(node);
            self.map.insert(key, (*self.list.front().unwrap()).clone());
        }
    }
}

//An LFU (Least Frequently Used) Cache removes the item with the lowest reference frequency. It also requires delete and insert operations to be O(1).

//TODO- Implement LFU Cache (Kinda hard using Rc and RefCells)

pub fn largest_rectangle_area(heights: Vec<i32>) -> i32 {
    let mut stack = VecDeque::new();
    let mut left_smaller_index = vec![0; heights.len()];
    let mut right_smaller_index = vec![0; heights.len()];

    for (ind, &ele) in heights.iter().enumerate() {
        while stack.back().map_or(false, |&i| heights[i] >= ele) {
            stack.pop_back();
        }

        left_smaller_index[ind] = stack.back().copied().map(|x| x as i32).unwrap_or(-1);
        stack.push_back(ind);
    }

    stack.clear();

    for (ind, &ele) in heights.iter().enumerate().rev() {
        while stack.back().map_or(false, |&i| heights[i] >= ele) {
            stack.pop_back();
        }

        right_smaller_index[ind] = stack.back().copied().map(|x| x as i32).unwrap_or(heights.len() as i32);
        stack.push_back(ind);
    }

    left_smaller_index.into_iter()
        .zip(right_smaller_index.into_iter())
        .zip(heights.into_iter())
        .map(|((left, right), height)| (right - left - 1) * height).max().unwrap()
}

pub fn largest_rectangle_area_optimised(heights: Vec<i32>) -> i32 {
    heights.iter().chain(&[0]).enumerate()
        .fold((vec![], 0), |(mut v, mut ans), (i, &x)| { //Vec acts like a stack
            while let Some(&y) = v.last() {
                if x > heights[y] {
                    break;
                }
                let height = heights[v.pop().unwrap()];
                let temp = if let Some(&i) = v.last() {
                    i as i32
                } else {
                    -1
                };
                let weight = i as i32 - temp - 1;
                ans = ans.max(height * weight);
            }
            v.push(i);
            (v, ans)
        }).1
}

pub fn max_sliding_window(nums: Vec<i32>, k: usize) -> Vec<i32> {
    //No need to explicitly check if dequeue.is_empty since the front will be None
    let mut dequeue = VecDeque::new();
    let mut ans = Vec::new();

    for (ind, &ele) in nums.iter().enumerate() {
        while dequeue.front().map_or(false, |&f| Some(f) == ind.checked_sub(ind)) { //Check for out of bounds
            dequeue.pop_front();
        }

        while dequeue.back().map_or(false, |&b| nums[b] < ele) { //Remove all elements smaller than current
            dequeue.pop_back();
        }

        dequeue.push_back(ind);

        if ind >= k - 1 { //After the first 2 elements, we can push to the answer
            ans.push(nums[*dequeue.front().unwrap()]);
        }
    }

    ans
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_stack() {
        let stack: Stack<i32, 5> = Stack::new();
        assert!(stack.top.is_none());
        assert!(stack.array.iter().all(|&x| x.is_none()));
    }

    #[test]
    fn test_push() {
        let mut stack: Stack<i32, 5> = Stack::new();
        assert_eq!(stack.push(1), Ok(()));
        assert_eq!(stack.top, Some(0));
        assert_eq!(stack.array[0], Some(1));
    }

    #[test]
    fn test_push_overflow() {
        let mut stack: Stack<i32, 1> = Stack::new();
        assert_eq!(stack.push(1), Ok(()));
        assert_eq!(stack.push(2), Err(StackError::Overflow));
    }

    #[test]
    fn test_pop() {
        let mut stack: Stack<i32, 5> = Stack::new();
        stack.push(1).unwrap();
        assert_eq!(stack.pop(), Ok(1));
        assert!(stack.top.is_none());
    }

    #[test]
    fn test_pop_underflow() {
        let mut stack: Stack<i32, 5> = Stack::new();
        assert_eq!(stack.pop(), Err(StackError::Underflow));
    }

    #[test]
    fn test_new_queue() {
        let queue: Queue<i32, 5> = Queue::new();
        assert!(queue.front.is_none());
        assert!(queue.rear.is_none());
        assert!(queue.array.iter().all(|&x| x.is_none()));
    }

    #[test]
    fn test_enqueue() {
        let mut queue: Queue<i32, 5> = Queue::new();
        assert_eq!(queue.enqueue(1), Ok(()));
        assert_eq!(queue.front, Some(0));
        assert_eq!(queue.rear, Some(1));
        assert_eq!(queue.array[0], Some(1));
    }

    #[test]
    fn test_enqueue_overflow() {
        let mut queue: Queue<i32, 1> = Queue::new();
        assert_eq!(queue.enqueue(1), Ok(()));
        assert_eq!(queue.enqueue(2), Err(QueueError::Overflow));
    }

    #[test]
    fn test_dequeue() {
        let mut queue: Queue<i32, 5> = Queue::new();
        queue.enqueue(1).unwrap();
        assert_eq!(queue.dequeue(), Ok(1));
        assert!(queue.front.is_none());
        assert!(queue.rear.is_none());
    }

    #[test]
    fn test_dequeue_underflow() {
        let mut queue: Queue<i32, 5> = Queue::new();
        assert_eq!(queue.dequeue(), Err(QueueError::Underflow));
    }

    #[test]
    fn valid_parenthesis_test() {
        assert_eq!(is_valid("()".to_string()), true);
        assert_eq!(is_valid("()[]{}".to_string()), true);
        assert_eq!(is_valid("(]".to_string()), false);
    }

    #[test]
    fn next_greater_element_two_test() {
        assert_eq!(next_greater_elements(vec![1, 2, 1]), vec![2, -1, 2]);
        assert_eq!(next_greater_elements(vec![1, 2, 3, 4, 3]), vec![2, 3, 4, -1, 4]);
    }

    #[test]
    fn lru_test() {
        let mut lru = LRUCache::new(2);
        lru.put(1, 1);
        lru.put(2, 2);
        lru.put(1, 3);
        assert_eq!(lru.get(1), Some(3));
        lru.put(3, 3);
        assert_eq!(lru.get(2), None);
        lru.put(4, 4);
        assert_eq!(lru.get(1), None);
    }

    #[test]
    fn largest_rectangle_area_test() {
        assert_eq!(largest_rectangle_area(vec![2, 1, 5, 6, 2, 3]), 10);
        assert_eq!(largest_rectangle_area(vec![2, 4]), 4);
    }

    #[test]
    fn largest_rectangle_area_test_again() {
        assert_eq!(largest_rectangle_area_optimised(vec![2, 1, 5, 6, 2, 3]), 10);
        assert_eq!(largest_rectangle_area_optimised(vec![2, 4]), 4);
    }

    #[test]
    fn sliding_window_max_test() {
        assert_eq!(max_sliding_window(vec![1, 3, -1, -3, 5, 3, 6, 7], 3), vec![3, 3, 5, 5, 6, 7]);
        assert_eq!(max_sliding_window(vec![1], 1), vec![1]);
    }
}
use std::cell::RefCell;
use std::cmp::{Ordering};
use std::collections::VecDeque;
use std::rc::Rc;

/* A tree is a non-linear data structure which shows hierarchical relation between elements. It is a DAG. Each edge defines a parent-child relationship between the nodes. There a unique path from the root to each node.
The root is the top most node of a tree (it has no parent)
The predecessor of a node is called its parent. The successor of a node is called its child (At most 2 for a binary tree)
The nodes with no children are leaf/external nodes. Thus, a node with at least one child is called an internal node
The number of children is the degree of the node. The degree of the tree is max(degree(nodes))
The depth of a tree is the number of edges in the path from root to the node. (Root has depth = 0)
The height of a tree is the number of edges in the longest path from the node to a leaf node. (Leaf nodes have height = 0)
 */

/* Binary trees are trees with at most 2 children for every node
A full/proper binary tree is one where each node has either 0 or 2 children
A complete binary tree is one where all tree levels are filled completely except the lowest level
A prefect binary tree is one where all levels are completely filled and all leaves are at the same level
A balanced binary tree is one where |height(left-sub-tree)-height(right-sub-tree)| <= 1
 */

/* A Binary Search Tree is an ordered binary tree. The value of the node is greater than the left node and lesser than the right node.
It gives O(log n) operations for insertion, deletion and searching
 */
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Node {
    pub value: i32,
    pub left: Option<Rc<RefCell<Node>>>,
    pub right: Option<Rc<RefCell<Node>>>,
    //Here we use Rc<RefCell> this allows us to have multiple owners (Rc) so that it can be shared among its parent and sibling nodes.
    //RefCell allows us to change the value of the left and right children after it is created, thus allowing interior mutability
}

///This is defined to easily convert a Node to Option<Rc<RefCell<Node>>>
impl Into<Option<Rc<RefCell<Node>>>> for Node {
    fn into(self) -> Option<Rc<RefCell<Node>>> {
        Some(Rc::new(RefCell::new(self)))
    }
}

impl Node {
    pub fn new(value: i32) -> Self {
        Node {
            value,
            left: None,
            right: None,
        }
    }

    pub fn add_left(&mut self, value: i32) {
        self.left = Node::new(value).into();
    }
    pub fn add_right(&mut self, value: i32) {
        self.right = Node::new(value).into();
    }

    pub fn insert(&mut self, value: i32) -> Result<(), &'static str> {
        match value.cmp(&self.value) {
            Ordering::Less => {
                if let Some(left) = &self.left {
                    left.borrow_mut().insert(value)
                } else {
                    self.left = Node::new(value).into();
                    Ok(())
                }
            }
            Ordering::Equal => Err("Value already exists in the BST"),
            Ordering::Greater => {
                if let Some(right) = &self.right {
                    right.borrow_mut().insert(value)
                } else {
                    self.right = Node::new(value).into();
                    Ok(())
                }
            }
        }
    }

    //Traversal algorithms for trees are majorly divided into Breadth First Search and Depth First Search (In-Order Traversal, Pre-Order Traversal and Post-Order Traversal)
    ///Left Root Right. This is in-order traversal using recursion
    pub fn inorder_vector(&self) -> Vec<i32> {
        let mut res = Vec::new();

        if let Some(left) = &self.left { //Left
            let mut left_part = left.borrow().inorder_vector();
            res.append(&mut left_part);
        }
        res.push(self.value); //Root
        if let Some(right) = &self.right { //Right
            let mut right_part = right.borrow().inorder_vector();
            res.append(&mut right_part);
        }

        res
    }
    //All other recursive traversals are done in the same way only changing the order of the code blocks
    ///Traverse each level first before moving to the next
    pub fn bfs_vector(&self) -> Vec<Vec<i32>> {
        let mut res: Vec<Vec<i32>> = Vec::new();
        let mut queue: VecDeque<Rc<RefCell<Node>>> = VecDeque::new();
        queue.push_back(Rc::new(RefCell::new(self.clone()))); // Clone self and wrap it in Rc<RefCell<_>>

        while !queue.is_empty() {
            let mut level: Vec<i32> = Vec::new();
            for _ in 0..queue.len() {
                let node_rc = queue.pop_front().unwrap();
                let node = node_rc.borrow();
                level.push(node.value);

                if let Some(left) = &node.left {
                    queue.push_back(Rc::clone(left));
                }
                if let Some(right) = &node.right {
                    queue.push_back(Rc::clone(right));
                }
            }
            res.push(level);
        }

        res
    }

    //Iterative Pre-Order, In-Order and Post-Order traversals
    ///Root Left Right
    pub fn preorder_vector(&self) -> Vec<i32> {
        let mut res = Vec::new();
        let mut stack = VecDeque::new(); //To use VecDeque as a stack, use push_back and pop_back functions
        stack.push_back(Rc::new(RefCell::new(self.clone())));

        while !stack.is_empty() {
            let node = stack.pop_back().unwrap();
            let node = node.borrow();
            res.push(node.value);
            if let Some(right) = &node.right {
                stack.push_back(right.clone());
            }
            if let Some(left) = &node.left {
                stack.push_back(left.clone());
            }
        }

        res
    }
    ///Iterative
    pub fn in_order_vector(&self) -> Vec<i32> {
        let mut res = Vec::new();
        let mut stack = VecDeque::new();
        let mut current = Some(Rc::new(RefCell::new(self.clone())));

        while !stack.is_empty() || current.is_some() {
            while let Some(node) = current { //Left
                stack.push_back(node.clone());
                current = node.borrow().left.clone();
            }

            let node = stack.pop_back().unwrap();
            let node = node.borrow();

            res.push(node.value); //Root

            current = node.right.clone();
        }

        res
    }
    //Post-Order iteratively is not as simple and not really required
}

pub fn max_depth(root: Option<Rc<RefCell<Node>>>) -> i32 {
    root.map_or(0, |node| {
        let left_height = max_depth(node.borrow().left.clone());
        let right_height = max_depth(node.borrow().right.clone());
        1 + std::cmp::max(left_height, right_height)
    })
}

pub fn is_balanced(root: Option<Rc<RefCell<Node>>>) -> bool {
    root.map_or(false, |node| {
        let left_height = max_depth(node.borrow().left.clone());
        let right_height = max_depth(node.borrow().right.clone());

        (left_height - right_height).abs() <= 1 && is_balanced(node.borrow().left.clone()) && is_balanced(node.borrow().right.clone())
    })
}

///Diameter is the length of the longest path between any two nodes (not necessarily through root)
pub fn diameter_of_binary_tree(root: Option<Rc<RefCell<Node>>>) -> i32 {
    fn helper(root: Option<Rc<RefCell<Node>>>, diameter: &mut i32) -> i32 {
        root.map_or(0, |node| {
            let left_height = helper(node.borrow().left.clone(), diameter);
            let right_height = helper(node.borrow().right.clone(), diameter);
            *diameter = std::cmp::max(*diameter, left_height + right_height);

            1 + std::cmp::max(left_height, right_height)
        })
    }
    let mut diameter = 0;
    helper(root.clone(), &mut diameter);
    diameter
}

pub fn max_path_sum(root: Option<Rc<RefCell<Node>>>) -> i32 {
    fn helper(root: Option<Rc<RefCell<Node>>>, max: &mut i32) -> i32 {
        root.map_or(0, |node| {
            let left_sum = helper(node.borrow().left.clone(), max);
            let right_sum = helper(node.borrow().right.clone(), max);
            *max = std::cmp::max(*max, left_sum + right_sum + node.borrow().value);

            std::cmp::max(0, node.borrow().value + std::cmp::max(left_sum, right_sum))
        })
    }

    let mut max = i32::MIN;
    helper(root.clone(), &mut max);
    max
}

pub fn is_same_tree(p: Option<Rc<RefCell<Node>>>, q: Option<Rc<RefCell<Node>>>) -> bool {
    match (p, q) {
        (Some(p), Some(q)) => {
            p.borrow().value == q.borrow().value
                && is_same_tree(p.borrow().left.clone(), q.borrow().left.clone())
                && is_same_tree(p.borrow().right.clone(), q.borrow().right.clone())
        }
        (None, None) => true,
        _ => false
    }
}

pub fn zigzag_level_order(root: Option<Rc<RefCell<Node>>>) -> Vec<Vec<i32>> {
    let mut result = Vec::new();
    if root.is_none() {
        return result;
    }

    let mut queue = VecDeque::new();
    queue.push_back(root.unwrap().clone());
    let mut left_to_right = true;

    while !queue.is_empty() {
        let size = queue.len();
        let mut level = vec![0; size];
        for i in 0..size {
            let node_rc = queue.pop_front().unwrap();
            let node = node_rc.borrow();
            let index = if left_to_right {
                i
            } else {
                size - i - 1
            };
            level[index] = node.value;

            if let Some(left) = node.left.clone() {
                queue.push_back(left);
            }
            if let Some(right) = node.right.clone() {
                queue.push_back(right);
            }
        }
        left_to_right = !left_to_right;
        result.push(level);
    }

    result
}

///Start from the leftmost nodes first and so on
pub fn vertical_traversal(root: Option<Rc<RefCell<Node>>>) -> Vec<Vec<i32>> {
    todo!()
}

pub fn is_symmetric(root: Option<Rc<RefCell<Node>>>) -> bool {
    fn helper(left: Option<Rc<RefCell<Node>>>, right: Option<Rc<RefCell<Node>>>) -> bool {
        match (left, right) {
            (Some(left), Some(right)) => {
                left.borrow().value == right.borrow().value
                    && helper(left.borrow().left.clone(), right.borrow().right.clone())
                    && helper(left.borrow().right.clone(), right.borrow().left.clone())
            }
            (None, None) => true,
            _ => false
        }
    }

    root.map_or(true, |node| helper(node.borrow().left.clone(), node.borrow().right.clone()))
}

pub fn get_path(root: Option<Rc<RefCell<Node>>>, node: i32) -> Vec<i32> {
    let mut path = Vec::new();
    fn helper(root: Option<Rc<RefCell<Node>>>, path: &mut Vec<i32>, node: i32) -> bool {
        root.map_or(false, |nde| {
            path.push(nde.borrow().value);

            if nde.borrow().value == node {
                return true;
            }

            if helper(nde.borrow().left.clone(), path, node) || helper(nde.borrow().right.clone(), path, node) {
                return true;
            }

            path.pop(); //Backtracking step

            false
        })
    }
    if root.is_some() {
        helper(root, &mut path, node);
    }

    path
}

pub fn get_path_to_leaf(root: Option<Rc<RefCell<Node>>>) -> Vec<String> {
    let mut paths = Vec::new();
    let mut path = Vec::new();

    fn helper(node: Option<Rc<RefCell<Node>>>, path: &mut Vec<i32>, paths: &mut Vec<Vec<i32>>) {
        if let Some(node) = node {
            path.push(node.borrow().value);
            if node.borrow().left.is_none() && node.borrow().right.is_none() {
                paths.push(path.clone());
            } else {
                helper(node.borrow().left.clone(), path, paths);
                helper(node.borrow().right.clone(), path, paths);
            }
            path.pop();
        }
    }

    helper(root, &mut path, &mut paths);
    fn format_paths(paths: Vec<Vec<i32>>) -> Vec<String> {
        paths.into_iter()
            .map(|path| path.into_iter()
                .map(|i| i.to_string()).collect::<Vec<String>>().join("->"))
            .collect()
    }

    format_paths(paths)
}

pub fn lowest_common_ancestor(root: Option<Rc<RefCell<Node>>>, p: Option<Rc<RefCell<Node>>>, q: Option<Rc<RefCell<Node>>>) -> Option<Rc<RefCell<Node>>> {
    root.and_then(|root_node| {
        if Rc::ptr_eq(&root_node, &p.clone().unwrap()) || Rc::ptr_eq(&root_node, &q.clone().unwrap()) {
            return Some(root_node);
        }
        let left = lowest_common_ancestor(root_node.borrow().left.clone(), p.clone(), q.clone());
        let right = lowest_common_ancestor(root_node.borrow().right.clone(), p.clone(), q.clone());

        match (left, right) {
            (Some(_), Some(_)) => Some(root_node),
            (Some(left_node), None) => Some(left_node),
            (None, Some(right_node)) => Some(right_node),
            (None, None) => None
        }
    })
}

///The number of nodes in a level between any two nodes
pub fn width_of_binary_tree(root: Option<Rc<RefCell<Node>>>) -> i32 {
    //Consider a level to be filled and then find the number of nodes that would have been
    //Thus, it can only be done in a level with more than 1 node
    root.map_or(0, |node| {
        let mut ans = 0;
        let mut queue = VecDeque::new();
        queue.push_back((node.clone(), 0));

        while !queue.is_empty() {
            let size = queue.len();
            let min = queue.front().unwrap().1;
            let (mut first, mut last) = (0, 0);
            for i in 0..size {
                let (node, id) = queue.pop_front().unwrap();
                let id = id - min;
                if i == 0 {
                    first = id;
                }
                if i == size - 1 {
                    last = id;
                }

                if let Some(left) = node.borrow().left.clone() {
                    queue.push_back((left, id * 2 + 1));
                };
                if let Some(right) = node.borrow().right.clone() {
                    queue.push_back((right, id * 2 + 2));
                };
            }
            ans = std::cmp::max(ans, last - first + 1);
        }

        ans
    })
}

///Make the tree to follow the property (left + right = value)

pub fn children_sum_property(node: Option<Rc<RefCell<Node>>>) {
    if let Some(ref node) = node {
        let mut child_sum = 0;
        if let Some(left) = &node.borrow().left {
            child_sum += left.borrow().value;
        }
        if let Some(right) = &node.borrow().right {
            child_sum += right.borrow().value;
        }

        let mut val = node.borrow_mut();
        if child_sum >= val.value {
            val.value = child_sum;
        } else {
            let val_value = val.value; // Store val.value in a temporary variable
            if let Some(left) = val.left.as_mut() {
                left.borrow_mut().value = val_value; // Use the temporary variable here
            }
            if let Some(right) = val.right.as_mut() {
                right.borrow_mut().value = val_value; // And here
            }
        }

        children_sum_property(val.left.clone());
        children_sum_property(val.right.clone());
        drop(val);

        let mut total = 0;
        if let Some(left) = &node.borrow().left {
            total += left.borrow().value;
        }
        if let Some(right) = &node.borrow().right {
            total += right.borrow().value;
        }

        dbg!(total);

        if node.borrow().left.is_some() || node.borrow().right.is_some() {
            node.borrow_mut().value = total;
        }
    }
}





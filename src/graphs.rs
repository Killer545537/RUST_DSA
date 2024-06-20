/*
A graph is defined as a group of vertices and edges (used to connect the vertices). Any two connected vertices are called adjacent nodes
The degree of a node is the number of edges falling on it. A self loop adds 2 to the degree. An isolated node has d(V) = 0

Types of Graphs are Finite, Infinite, Trivial, Simple, Multi, NULL, Complete, Pseudo, Regular, Labelled, Directed, Bipartite
A null graph is one with more than one vertex but no edges
A graph which contains both parallel edges and self-loops are called pseudo graphs
A graph where each edge has a weight or any information is called a labelled graph

REPRESENTATION OF GRAPHS
Adjacency Matrix :- In this if there is an edge between a and b, mark matrix[a][b] = matrix[b][a] = 1 (Or the weight of the edge). This uses O(n*m) space
Adjacency List :- In this the ith index stores the vertices connected to i Uses O(2E) space. It is the most common and efficient for sparse graphs
The drawbacks of the above are when we use Sparse Matrices (where 0s are more), it wastes a lot of memory, thus, we use Linked Lists to save memory
 */

use std::cmp::{Ordering, Reverse};
use std::collections::{HashSet, VecDeque, HashMap, BinaryHeap};
use std::ops::Index;
use itertools::Itertools;

//TRAVERSAL TECHNIQUES (A graph can either start from 0 or 1, I will start my graphs from 1 unless the question states otherwise)
// Breadth-First-Search (BFS) uses a queue (can use recursion but basically the same algorithm, thus not worth it). T.C. = O(V+E) & S.C. = O(V)
pub fn breadth_first_search(adj_list: Vec<Vec<usize>>) -> Vec<usize> {
    //A vis(ited) array stores the nodes which have already been visited
    let mut vis = vec![false; adj_list.len() + 1]; //+1 because of 1-based indexing
    let mut queue = VecDeque::new();
    let mut bfs = Vec::new();

    vis[1] = true; //Mark the first node as visited
    queue.push_back(1);

    while let Some(node) = queue.pop_front() {
        bfs.push(node);

        for &adj_node in adj_list[node].iter() {
            if !vis[adj_node] {
                vis[adj_node] = true;
                queue.push_back(adj_node);
            }
        }
    }

    bfs
}

//Depth-First-Search (DFS) uses either recursion or stack. T.C. = O(V+2E) (undirected) O(V+E) (directed) when using adjacency list. But when E≅V^2, matrix has better time complexity of O(V^2) & S.C. = O(V)
pub fn depth_first_search_recursive(adj_list: Vec<Vec<usize>>) -> Vec<usize> {
    let mut vis = vec![false; adj_list.len() + 1];
    let mut dfs = Vec::new();
    fn helper(node: usize, adj_list: &Vec<Vec<usize>>, vis: &mut Vec<bool>, dfs: &mut Vec<usize>) {
        vis[node] = true;
        dfs.push(node);

        for &adj_node in adj_list[node].iter() {
            if !vis[adj_node] {
                helper(adj_node, adj_list, vis, dfs);
            }
        }
    }

    helper(1, &adj_list, &mut vis, &mut dfs);

    dfs
}

pub fn depth_first_search_iterative(adj_list: Vec<Vec<usize>>) -> Vec<usize> {
    let mut vis = vec![false; adj_list.len() + 1];
    let mut stack = VecDeque::new();
    let mut dfs = Vec::new();
    stack.push_back(1);

    while let Some(node) = stack.pop_back() {
        if !vis[node] {
            vis[node] = true;
            dfs.push(node);

            for &adj_node in adj_list[node].iter().rev() {
                if !vis[adj_node] {
                    stack.push_back(adj_node);
                }
            }
        }
    }

    dfs
}

//NOTE: In most of these problems we can use both BFS and DFS

///A province is a component of a graph which is not connected to any other component
pub fn number_of_provinces(adj_matrix: Vec<Vec<i32>>) -> i32 {
    let mut vis = vec![false; adj_matrix.len() + 1];
    let mut provinces = 0;
    //Here we can use any traversal technique
    fn helper(node: usize, vis: &mut Vec<bool>, adj_matrix: &Vec<Vec<i32>>) { //DFS traversal
        vis[node] = true;
        for adj_node in 0..adj_matrix.len() {
            if adj_matrix[node][adj_node] == 1 && !vis[adj_node] {
                helper(adj_node, vis, adj_matrix);
            }
        }
    }

    for node in 0..adj_matrix.len() {
        if !vis[node] {
            helper(node, &mut vis, &adj_matrix);
            provinces += 1;
        }
    }

    provinces
}

///Starting from (sr, sc), every node connected 4-directionally to it with the same color, color it with 'color' and so on
pub fn flood_fill(image: Vec<Vec<i32>>, sr: usize, sc: usize, color: i32) -> Vec<Vec<i32>> {
    //Simply, we just need to color all the connected nodes with the same initial color
    let initial_color = image[sr][sc];
    let mut final_image = image.clone();
    fn helper(row: usize, col: usize, image: &Vec<Vec<i32>>, final_image: &mut Vec<Vec<i32>>, initial_color: i32, color: i32) {
        final_image[row][col] = color;
        const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

        for direction in DIRECTIONS {
            let new_row = row as i32 + direction.0;
            let new_col = col as i32 + direction.1;

            if new_row >= 0 && new_row < image.len() as i32 && new_col >= 0 && new_col < image[0].len() as i32
                && image[new_row as usize][new_col as usize] == initial_color && final_image[new_row as usize][new_col as usize] != color
            {
                helper(new_row as usize, new_col as usize, image, final_image, initial_color, color);
            }
        }
    }

    helper(sr, sc, &image, &mut final_image, initial_color, color);
    final_image
}

///Return the time taken to rot all oranges. Any fresh orange in cell adjacent (4-directions) to rotten becomes rotten. //0 is empty, 1 is fresh, 2 is rotten
pub fn oranges_rotting(grid: Vec<Vec<i32>>) -> Option<i32> {
    let n = grid.len();
    let m = grid[0].len();
    let mut minutes = 0;
    //BFS Traversal
    let mut queue = VecDeque::new(); //(Row, Col), Time
    let mut vis = vec![vec![0; m]; n];
    const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for row in 0..n {
        for col in 0..m {
            if grid[row][col] == 2 {
                queue.push_back(((row, col), 0));
                vis[row][col] = 2;
            }
        }
    }

    while let Some(((row, col), time)) = queue.pop_front() {
        minutes = minutes.max(time);

        for direction in DIRECTIONS {
            let new_row = row as i32 + direction.0;
            let new_col = col as i32 + direction.1;

            if new_row >= 0 && new_row < n as i32 && new_col >= 0 && new_col < m as i32 //If the index is valid
                && grid[new_row as usize][new_col as usize] == 1 //If the orange is fresh
                && vis[new_row as usize][new_col as usize] != 2 //If the orange has not yet been rotten
            {
                queue.push_back(((new_row as usize, new_col as usize), time + 1));
                vis[new_row as usize][new_col as usize] = 2;
            }
        }
    }

    //If there is some orange which cannot be rotten
    for row in 0..n {
        for col in 0..m {
            if vis[row][col] != 2 && grid[row][col] == 1 {
                return None;
            }
        }
    }

    Some(minutes)
}

pub fn cycle_detection_bfs(adj_list: Vec<Vec<usize>>) -> bool {
    let mut vis = vec![false; adj_list.len() + 1];
    fn helper(node: usize, vis: &mut Vec<bool>, adj_list: &Vec<Vec<usize>>) -> bool {
        vis[node] = true;
        let mut queue = VecDeque::new();
        queue.push_back((node, None)); //-1 means it has no parent

        while let Some((node, parent)) = queue.pop_front() {
            for &adj_node in adj_list[node].iter() {
                if !vis[adj_node] {
                    vis[adj_node] = true;
                    queue.push_back((adj_node, Some(node)));
                } else if Some(adj_node) != parent {
                    //If the node is visited and not the parent of the current node
                    return true;
                }
            }
        }

        false
    }

    for i in 1..adj_list.len() { //To be able to traverse despite non-connected components
        if !vis[i] {
            if helper(i, &mut vis, &adj_list) {
                return true;
            };
        }
    }

    false
}

pub fn cycle_detection_dfs(adj_list: Vec<Vec<usize>>) -> bool {
    let mut vis = vec![false; adj_list.len() + 1];
    fn helper(node: usize, parent: Option<usize>, vis: &mut Vec<bool>, adj_list: &Vec<Vec<usize>>) -> bool {
        vis[node] = true;

        for &adj_node in adj_list[node].iter() {
            if !vis[adj_node] {
                if helper(adj_node, Some(node), vis, adj_list) {
                    return true;
                }
            } else if Some(adj_node) != parent {
                return true;
            }
        }

        false
    }

    for i in 1..adj_list.len() {
        if !vis[i] {
            if helper(i, None, &mut vis, &adj_list) {
                return true;
            }
        }
    }

    false
}

///Find the distance to the nearest 0
pub fn update_matrix(mat: Vec<Vec<i32>>) -> Vec<Vec<i32>> { //This is a multi-source BFS problem
    //If the cell itself is 0, the distance is 0, the distance between adjacent cells is 1
    let mut distance = vec![vec![0; mat[0].len()]; mat.len()];
    let mut vis = vec![vec![false; mat[0].len()]; mat.len()];
    let mut queue = VecDeque::new(); // (Row, Col), Distance
    const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for row in 0..mat.len() {
        for col in 0..mat[0].len() {
            if mat[row][col] == 0 {
                queue.push_back(((row, col), 0));
                vis[row][col] = true;
            }
        }
    }
    //BFS
    while let Some(((row, col), step)) = queue.pop_front() {
        distance[row][col] = step;

        for direction in DIRECTIONS {
            let new_row = row as i32 + direction.0;
            let new_col = col as i32 + direction.1;

            if new_row >= 0 && new_row < mat.len() as i32 && new_col >= 0 && new_col < mat[0].len() as i32
                && !vis[new_row as usize][new_col as usize] {
                vis[new_row as usize][new_col as usize] = true;
                queue.push_back(((new_row as usize, new_col as usize), step + 1));
            }
        }
    }

    distance
}

///Replace any region (of Os) surrounded by Xs on all four sides with X
pub fn surround_regions(board: &mut Vec<Vec<char>>) {
    //Regions on the boundaries cannot be changed (since they are not 'surrounded')
    //Thus, all Os connected to a boundary O will not be changed and all other will be converted to X
    let mut vis = vec![vec![false; board[0].len()]; board.len()];
    let is_border = |row, col| {
        row == 0 || row == board.len() - 1 || col == 0 || col == board[0].len() - 1
    };
    fn helper(row: usize, col: usize, board: &Vec<Vec<char>>, vis: &mut Vec<Vec<bool>>) {
        vis[row][col] = true;
        const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

        for direction in DIRECTIONS {
            let new_row = row as i32 + direction.0;
            let new_col = col as i32 + direction.1;

            if new_row >= 0 && new_row < board.len() as i32 && new_col >= 0 && new_col < board[0].len() as i32
                && !vis[new_row as usize][new_col as usize] && board[new_row as usize][new_col as usize] == 'O' {
                helper(new_row as usize, new_col as usize, board, vis);
            }
        }
    }

    for row in 0..board.len() {
        for col in 0..board[0].len() {
            if is_border(row, col) && board[row][col] == 'O' {
                helper(row, col, board, &mut vis);
            }
        }
    }

    for row in 0..board.len() {
        for col in 0..board[0].len() {
            if !vis[row][col] && board[row][col] == 'O' {
                board[row][col] = 'X';
            }
        }
    }
}

///We need to find the number of land regions (1) from where we cannot walk out of the boundary
pub fn num_enclaves(grid: Vec<Vec<i32>>) -> usize {
    //Any 1 connected to the boundary will not be counted in out answer
    let mut vis = vec![vec![false; grid[0].len()]; grid.len()];
    let mut queue = VecDeque::new(); //Using BFS

    for row in 0..grid.len() {
        for col in 0..grid[0].len() {
            if row == 0 || col == 0 || row == grid.len() - 1 || col == grid[0].len() - 1 {
                if grid[row][col] == 1 {
                    vis[row][col] = true;
                    queue.push_back((row, col));
                }
            }
        }
    }

    const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    while let Some((row, col)) = queue.pop_front() {
        for direction in DIRECTIONS {
            let new_row = row as i32 + direction.0;
            let new_col = col as i32 + direction.1;

            if new_row >= 0 && new_row < grid.len() as i32 && new_col >= 0 && new_col < grid[0].len() as i32
                && !vis[new_row as usize][new_col as usize] && grid[new_row as usize][new_col as usize] == 1 {
                vis[new_row as usize][new_col as usize] = true;
                queue.push_back((new_row as usize, new_col as usize));
            }
        }
    }


    grid.iter()
        .zip(vis.iter())
        .flat_map(|(g_row, v_row)| g_row.iter().zip(v_row.iter()))
        .filter(|(&g, &v)| g == 1 && !v)
        .count()
}

//TODO- Find a way to implement "Number of Distinct Islands in Rust

//A bipartite graph is one which can be colored with 2 colors such that no adjacent nodes are the same color
//Precisely, if the vertex V can be partitioned into sets X and Y such that all edges have one end in X and the other in Y
//Any linear graph with no cycle is always bipartite. Any graph with an even cycle length is also bipartite
pub fn is_bipartite_bfs(grid: Vec<Vec<usize>>) -> bool { //The graph is 0-indexed
    let mut queue = VecDeque::new();
    queue.push_back(0);
    let mut color = vec![None; grid.len()];
    color[0] = Some(true);

    for i in 0..grid.len() { //Some BS because of non-connected components
        if color[i].is_none() {
            queue.push_back(i);
            color[i] = Some(true);
        }

        while let Some(node) = queue.pop_front() {
            for &adj_node in grid[node].iter() {
                if color[adj_node].is_none() { //If the node is not colored
                    queue.push_back(adj_node);
                    color[adj_node] = Some(!color[node].unwrap());
                } else if color[adj_node] == color[node] { //If the node is the same color
                    return false;
                }
            }
        }
    }

    true
}

pub fn is_bipartite_dfs(grid: Vec<Vec<usize>>) -> bool {
    let mut color = vec![None; grid.len()];
    fn helper(node: usize, col: bool, color: &mut Vec<Option<bool>>, grid: &Vec<Vec<usize>>) -> bool {
        color[node] = Some(col);

        for &adj_node in grid[node].iter() {
            if color[adj_node].is_none() {
                if !helper(adj_node, !col, color, grid) {
                    return false;
                };
            } else if color[adj_node] == color[node] {
                return false;
            }
        }

        true
    }

    for i in 0..grid.len() {
        if color[i].is_none() {
            if !helper(i, false, &mut color, &grid) {
                return false;
            };
        }
    }

    true
}

pub fn cycle_directed(adj_list: Vec<Vec<usize>>) -> bool {
    let mut vis = vec![false; adj_list.len()];
    let mut path_vis = vec![false; adj_list.len()];
    fn helper(node: usize, adj_list: &Vec<Vec<usize>>, vis: &mut Vec<bool>, path_vis: &mut Vec<bool>) -> bool { //DFS Traversal
        vis[node] = true;
        path_vis[node] = true;

        for &adj_node in adj_list[node].iter() {
            if !vis[adj_node] {
                if helper(adj_node, adj_list, vis, path_vis) {
                    return true;
                }
            } else if path_vis[adj_node] { //If the node has been visited on the same path
                return true;
            }
        }

        path_vis[node] = false; //Backtracking
        false
    }

    for i in 0..adj_list.len() {
        if !vis[i] {
            if helper(i, &adj_list, &mut vis, &mut path_vis) {
                return true;
            }
        }
    }

    false
}

///A safe node is one form where all edges point to a terminal node (no outgoing edges)
pub fn eventual_safe_nodes(graph: Vec<Vec<usize>>) -> Vec<usize> {
    //
    let mut vis = vec![false; graph.len()];
    let mut path_vis = vec![false; graph.len()];
    let mut check = vec![false; graph.len()];
    fn helper(node: usize, graph: &Vec<Vec<usize>>, vis: &mut Vec<bool>, path_vis: &mut Vec<bool>, check: &mut Vec<bool>) -> bool {
        vis[node] = true;
        path_vis[node] = true;
        check[node] = false;

        for &adj_node in &graph[node] {
            if !vis[adj_node] {
                if helper(adj_node, graph, vis, path_vis, check) {
                    return true;
                }
            } else if path_vis[adj_node] {
                return true;
            }
        }

        check[node] = true; // If the node is not a part of a cycle it is eventually safe
        path_vis[node] = false; //Backtracking
        false
    }

    for i in 0..graph.len() {
        if !vis[i] {
            helper(i, &graph, &mut vis, &mut path_vis, &mut check);
        }
    }

    let mut safe_nodes = Vec::new();
    for i in 0..graph.len() {
        if check[i] {
            safe_nodes.push(i);
        }
    }

    safe_nodes
}

//Topological Sorting :- It is a linear ordering of vertices such that v appears before w if there is an edge between v->w. There may be multiple solutions. It can only be done for DAG
pub fn topological_sort(adj_list: Vec<Vec<usize>>) -> Vec<usize> {
    let mut vis = vec![false; adj_list.len()];
    let mut stack = VecDeque::new();
    fn helper(node: usize, adj_list: &Vec<Vec<usize>>, vis: &mut Vec<bool>, stack: &mut VecDeque<usize>) { //DFS Traversal
        vis[node] = true;

        for &adj_node in adj_list[node].iter() {
            if !vis[adj_node] {
                helper(adj_node, adj_list, vis, stack);
            }
        }

        stack.push_back(node);
    }

    for i in 1..adj_list.len() {
        if !vis[i] {
            helper(i, &adj_list, &mut vis, &mut stack);
        }
    }

    stack.into_iter().rev().collect()
}

//Now finding the topological sort using BFS is called Kahn's Algorithm
///T.C. = O(V+E)
pub fn kahn_algorithm(adj_list: Vec<Vec<usize>>) -> Vec<usize> {
    let mut indegree = vec![0; adj_list.len()]; //The number of edges coming into a node
    let mut queue = VecDeque::new();
    for node in 0..adj_list.len() {
        for &adj_node in &adj_list[node] {
            indegree[adj_node] += 1;
        }
    }
    //It is guaranteed that at least one node has indegree 0
    for node in 0..adj_list.len() {
        if indegree[node] == 0 {
            queue.push_back(node);
        }
    }

    let mut topo = Vec::new();
    while let Some(node) = queue.pop_front() {
        topo.push(node);
        //Remove the node from the indegree
        for &adj_node in &adj_list[node] {
            indegree[adj_node] -= 1;
            if indegree[adj_node] == 0 {
                queue.push_back(adj_node);
            }
        }
    }

    topo
}

pub fn cycle_directed_topological(adj_list: Vec<Vec<usize>>) -> bool {
    //Since Kahn's Algorithm is only applicable for DAG, using it on a graph with a cycle will not return the topo-sort of size V
    let mut indegree = vec![0; adj_list.len()];
    let mut queue = VecDeque::new();
    for node in 0..adj_list.len() {
        for &adj_node in &adj_list[node] {
            indegree[adj_node] += 1;
        }
    }

    for node in 0..adj_list.len() {
        if indegree[node] == 0 {
            queue.push_back(node);
        }
    }
    let mut topo = Vec::new();
    while let Some(node) = queue.pop_front() {
        topo.push(node);

        for &adj_node in &adj_list[node] {
            indegree[adj_node] -= 1;
            if indegree[adj_node] == 0 {
                queue.push_back(adj_node);
            }
        }
    }

    topo.len() != adj_list.len()
}

///prerequisites[i] = (a, b) which means we need to complete course b to take course a (b should be done before a)
pub fn can_finish(num_courses: usize, prerequisites: Vec<(usize, usize)>) -> bool {
    //Courses cannot be finished iff there is a cycle in the task graph
    let mut graph = vec![Vec::new(); num_courses]; //Creating the tasks graph
    for &(a, b) in &prerequisites {
        graph[b].push(a);
    }
    //Using Kahn's Algorithm
    let mut indegree = vec![0; num_courses];
    let mut queue = VecDeque::new();
    for node in 0..num_courses {
        for &adj_node in &graph[node] {
            indegree[adj_node] += 1;
        }
    }
    for node in 0..num_courses {
        if indegree[node] == 0 {
            queue.push_back(node);
        }
    }

    let mut topo = Vec::new();
    while let Some(node) = queue.pop_front() {
        topo.push(node);

        for &adj_node in &graph[node] {
            indegree[adj_node] -= 1;
            if indegree[adj_node] == 0 {
                queue.push_back(adj_node);
            }
        }
    }

    topo.len() == num_courses
}

pub fn eventual_safe_nodes_bfs(graph: Vec<Vec<usize>>) -> Vec<usize> {
    //Here, we will use Kahn's Algorithm to find the safe nodes
    let mut reversed_graph = vec![Vec::new(); graph.len()];
    let mut indegree = vec![0; graph.len()];
    let mut queue = VecDeque::new();
    for node in 0..graph.len() {
        for &adj_node in &graph[node] {
            reversed_graph[adj_node].push(node);
            indegree[node] += 1;
        }
    }
    for node in 0..graph.len() {
        if indegree[node] == 0 {
            queue.push_back(node);
        }
    }

    let mut safe_nodes = Vec::new();
    while let Some(node) = queue.pop_front() {
        safe_nodes.push(node);
        for &adj_node in &reversed_graph[node] {
            indegree[adj_node] -= 1;
            if indegree[adj_node] == 0 {
                queue.push_back(adj_node);
            }
        }
    }

    safe_nodes.sort();
    safe_nodes
}

///The alien dictionary has the given order using the first 'alphabets' of the English language. We need to find the order of the new alphabets
pub fn alien_dictionary(dictionary: Vec<String>, alphabets: usize) -> String {
    let mut graph = vec![Vec::new(); alphabets];
    let char_to_usize = |c: char| -> usize {
        c as usize - 'a' as usize
    };
    let usize_to_letter = |u: usize| {
        (u + 'a' as usize) as u8 as char
    };

    for (s1, s2) in dictionary.iter().zip(dictionary.iter().skip(1)) { //Compare two adjacent words
        for i in 0..std::cmp::min(s1.len(), s2.len()) {
            let s1_ith = s1.chars().nth(i).unwrap();
            let s2_ith = s2.chars().nth(i).unwrap();
            if s1_ith != s2_ith { //Find the first difference (This is the reason s1 is before s2 in the dict)
                graph[char_to_usize(s1_ith)].push(char_to_usize(s2_ith));
                break;
            }
        }
    }
    //Now the graph is created
    //Perform Topological Sort and convert it back to a String
    kahn_algorithm(graph).iter().map(|&u| usize_to_letter(u)).collect()
}

pub fn shortest_path_in_dag(adj_list: Vec<Vec<(usize, i32)>>) -> Vec<Option<i32>> { //Vec<Vec<(adj_node, distance)>>
    //Find the topological sort (DFS is better since we will get the stack)
    pub fn topological_dfs(adj_list: &Vec<Vec<(usize, i32)>>) -> VecDeque<usize> {
        let mut vis = vec![false; adj_list.len()];
        let mut stack = VecDeque::new();
        fn helper(node: usize, adj_list: &Vec<Vec<(usize, i32)>>, vis: &mut Vec<bool>, stack: &mut VecDeque<usize>) { //DFS Traversal
            vis[node] = true;

            for &(adj_node, _) in &adj_list[node] {
                if !vis[adj_node] {
                    helper(adj_node, adj_list, vis, stack);
                }
            }

            stack.push_back(node);
        }

        for i in 0..adj_list.len() {
            if !vis[i] {
                helper(i, &adj_list, &mut vis, &mut stack);
            }
        }

        stack
    }

    let mut stack = topological_dfs(&adj_list);
    let mut distance = vec![None; adj_list.len()];
    distance[0] = Some(0); //Distance from itself is 0

    while let Some(node) = stack.pop_back() {
        if let Some(dist) = distance[node] {
            for &(adj_node, len) in &adj_list[node] {
                let new_distance = dist + len;
                distance[adj_node] = match distance[adj_node] {
                    Some(current_distance) => Some(current_distance.min(new_distance)),
                    None => Some(new_distance),
                };
            }
        }
    }

    distance
}

///The graph is an undirected graph with each edge weight = 1
pub fn shortest_path_with_unit_weights(adj_list: Vec<Vec<usize>>, source: usize) -> Vec<i32> {
    let mut distance = vec![i32::MAX; adj_list.len()];
    let mut queue = VecDeque::new();
    distance[source] = 0;
    queue.push_back(source);

    while let Some(node) = queue.pop_front() {
        for &adj_node in &adj_list[node] {
            if distance[node] + 1 < distance[adj_node] {
                distance[adj_node] = distance[node] + 1;
                queue.push_back(adj_node);
            }
        }
    }

    distance
}

///In each move, change one letter such that new word is in word_list
pub fn ladder_length(begin_word: String, end_word: String, word_list: Vec<String>) -> i32 {
    let mut queue = VecDeque::new();
    queue.push_back((begin_word.clone(), 1));
    let mut set: HashSet<String> = word_list.clone().into_iter().collect();
    set.remove(&begin_word);

    while let Some((word, steps)) = queue.pop_front() {
        if word == end_word {
            return steps;
        }
        for i in 0..word.len() {
            let mut word_vec: Vec<char> = word.chars().collect();
            for c in 'a'..='z' {
                word_vec[i] = c;
                let new_word: String = word_vec.iter().collect();

                if set.contains(&new_word) {
                    set.remove(&new_word);
                    queue.push_back((new_word, steps + 1));
                }
            }
        }
    }

    0
}

//This will result in TLE
pub fn find_ladder(begin_word: String, end_word: String, word_list: Vec<String>) -> Vec<Vec<String>> {
    let mut set: HashSet<String> = word_list.clone().into_iter().collect();
    let mut queue = VecDeque::new();
    queue.push_back(vec![begin_word.clone()]);
    let mut used_on_level = vec![begin_word.clone()];
    let mut level = 0;
    let mut ans = Vec::new();

    while let Some(mut path) = queue.pop_front() {
        if path.len() > level {
            level += 1;
            for i in &used_on_level {
                set.remove(i);
            }
        }

        let word = path.last().unwrap().clone();

        if word == end_word {
            if ans.len() == 0 {
                ans.push(path.clone());
            } else if ans[0].len() == path.len() {
                ans.push(path.clone());
            }
        }

        for i in 0..word.len() {
            let mut word_vec: Vec<char> = word.chars().collect();
            for c in 'a'..='z' {
                word_vec[i] = c;
                if set.contains(&word_vec.iter().collect::<String>()) {
                    path.push(word_vec.iter().collect());
                    queue.push_back(path.clone());
                    used_on_level.push(word_vec.iter().collect());
                    path.pop();
                }
            }
        }
    }

    ans
}

pub fn find_ladder_optimised(begin_word: String, end_word: String, word_list: Vec<String>) -> Vec<Vec<String>> {
    let mut word_level = HashMap::new();
    let word_length = begin_word.len();
    let mut set: HashSet<String> = word_list.into_iter().collect();
    let mut queue = VecDeque::new();
    queue.push_back(begin_word.clone());
    word_level.insert(begin_word.clone(), 1);
    set.remove(&begin_word);
    fn helper(word: String, begin_word: String, path: &mut Vec<String>, ans: &mut Vec<Vec<String>>, word_level: &HashMap<String, i32>) {
        if word == begin_word {
            ans.push(path.clone().into_iter().rev().collect());
            return;
        }
        let steps = *word_level.get(&word).unwrap();
        for i in 0..word.len() {
            for c in 'a'..='z' {
                let mut word_vec: Vec<char> = word.chars().collect();
                word_vec[i] = c;
                let new_word: String = word_vec.iter().collect();

                if let Some(level) = word_level.get(&new_word) {
                    if level + 1 == steps {
                        path.push(new_word.clone());
                        helper(new_word.clone(), begin_word.clone(), path, ans, word_level);
                        path.pop();
                    }
                }
            }
        }
    }

    while let Some(word) = queue.pop_front() {
        let step = *word_level.get(&word).unwrap();

        if word == end_word {
            break;
        }

        for i in 0..word_length {
            let mut word_vec: Vec<char> = word.chars().collect();

            for c in 'a'..='z' {
                word_vec[i] = c;
                let new_word: String = word_vec.iter().collect();
                if set.contains(&new_word) {
                    queue.push_back(new_word.clone());
                    set.remove(&new_word);
                    word_level.insert(new_word, step + 1);
                }
            }
        }
    }

    let mut ans = Vec::new();
    if word_level.contains_key(&end_word) {
        let mut path = Vec::new();
        path.push(end_word.clone());
        helper(end_word.clone(), begin_word.clone(), &mut path, &mut ans, &word_level);
    }

    ans
}

///A min-heap has the minimum element as the root. T.C. = O(E log V). This works on positive weighted acyclic graphs
pub fn dijkstra_algorithm_min_heap(adj_list: Vec<Vec<(usize, i32)>>, source: usize) -> Vec<Option<i32>> {
    //In Rust a Binary Heap is by-default a max-heap. Reverse(T) is used to reverse the ordering
    let mut heap: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::new(); //Min-Heap/Priority Queue in C++
    let mut distances = vec![None; adj_list.len()];
    distances[source] = Some(0);
    heap.push(Reverse((0, source)));

    while let Some(Reverse((distance, node))) = heap.pop() {
        for &(adj_node, edge_length) in &adj_list[node] {
            distances[adj_node] = match distances[adj_node] {
                None => { //If the node has not been reached push it into the heap and update the distance
                    heap.push(Reverse((distance + edge_length, adj_node)));
                    Some(distance + edge_length)
                }
                Some(current_length) => {
                    if distance + edge_length < current_length { //If the new distance if better, push into the heap
                        heap.push(Reverse((distance + edge_length, adj_node)));
                        Some(distance + edge_length)
                    } else {
                        Some(current_length)
                    }
                }
            };
        }
    }

    distances
}
//The set approach will not work, since C++'s set is also a min-heap type data structure while HashSet does not guarantee any order

///Find the path from 1 to N
pub fn shortest_path(adj_list: Vec<Vec<(usize, i32)>>) -> Option<Vec<usize>> { //This problem uses 1-based indexing
    //We will store from where each node is reached from
    let mut parents: Vec<usize> = (0..adj_list.len()).collect();
    let mut distances = vec![None; adj_list.len()];
    distances[1] = Some(0);
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((0, 1)));

    while let Some(Reverse((distance, node))) = heap.pop() {
        for &(adj_node, len) in &adj_list[node] {
            distances[adj_node] = match distances[adj_node] {
                None => {
                    heap.push(Reverse((distance + len, adj_node)));
                    parents[adj_node] = node;
                    Some(distance + len)
                }
                Some(current_length) => {
                    if distance + len < current_length {
                        heap.push(Reverse((distance + len, adj_node)));
                        parents[adj_node] = node;
                        Some(distance + len)
                    } else {
                        Some(current_length)
                    }
                }
            }
        }
    }

    if distances[adj_list.len() - 1].is_none() {
        return None;
    }
    let mut path = Vec::new();
    let mut node = adj_list.len() - 1;
    while parents[node] != node {
        path.push(node);
        node = parents[node];
    }
    path.push(1);
    path.reverse();

    Some(path)
}

///Find the length of the shortest path from (0, 0) to (n-1, n-1), traveling only on 0s (8-directional movement)
pub fn shortest_path_binary_matrix(grid: Vec<Vec<i32>>) -> i32 {
    let n = grid.len();
    if grid[0][0] == 1 || grid[n - 1][n - 1] == 1 {
        return -1;
    }
    if n == 1 { //For the test case [[0]]
        return 1;
    }
    const DIRECTIONS: [(i32, i32); 8] = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 1), (1, -1)];
    let mut queue = VecDeque::new();
    let mut distances = vec![vec![i32::MAX; n]; n];
    queue.push_back((1, (0, 0)));
    distances[0][0] = 1;

    while let Some((distance, (row, col))) = queue.pop_front() {
        for direction in DIRECTIONS {
            let new_row = row as i32 + direction.0;
            let new_col = col as i32 + direction.1;

            if new_row >= 0 && new_row < n as i32 && new_col >= 0 && new_col < n as i32
                && grid[new_row as usize][new_col as usize] == 0
                && distance + 1 < distances[new_row as usize][new_col as usize] {
                distances[new_row as usize][new_col as usize] = 1 + distance;
                if new_row == n as i32 - 1 && new_col == n as i32 - 1 {
                    return 1 + distance;
                }
                queue.push_back((1 + distance, (new_row as usize, new_col as usize)));
            }
        }
    }

    -1
}

///The effort is the absolute difference in the cell values
pub fn minimum_effort_path(heights: Vec<Vec<i32>>) -> i32 {
    let (rows, columns) = (heights.len(), heights[0].len());
    let mut heap = BinaryHeap::new(); //Difference, Row, Col
    heap.push(Reverse((0, (0, 0))));
    let mut differences = vec![vec![i32::MAX; columns]; rows];
    differences[0][0] = 0;
    const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    while let Some(Reverse((effort, (row, col)))) = heap.pop() {
        if row == rows - 1 && col == columns - 1 {
            return effort;
        }
        for direction in DIRECTIONS {
            let new_row = row as i32 + direction.0;
            let new_col = col as i32 + direction.1;

            if new_row >= 0 && new_row < rows as i32 && new_col >= 0 && new_col < columns as i32 {
                let new_row = new_row as usize;
                let new_col = new_col as usize;
                let new_effort = std::cmp::max(
                    (heights[new_row][new_col] - heights[row][col]).abs(),
                    effort,
                );

                if new_effort < differences[new_row][new_col] {
                    differences[new_row][new_col] = new_effort;
                    heap.push(Reverse((new_effort, (new_row, new_col))));
                }
            }
        }
    }
    unreachable!()
}

pub fn find_cheapest_price(cities: i32, flights: Vec<Vec<(usize, i32)>>, src: usize, dst: usize, max_stops: i32) -> Option<i32> {
    let mut distances = vec![i32::MAX; cities as usize];
    distances[src] = 0;
    let mut queue = VecDeque::new(); //Stops, Node, Cost
    queue.push_back((0, (src, 0)));

    while let Some((stops, (node, cost))) = queue.pop_front() {
        if stops > max_stops {
            continue;
        }

        for &(adj_node, edge_cost) in &flights[node] {
            if cost + edge_cost < distances[adj_node] && stops <= max_stops {
                distances[adj_node] = cost + edge_cost;
                queue.push_back((stops + 1, (adj_node, cost + edge_cost)));
            }
        }
    }

    return if distances[dst] == i32::MAX {
        None
    } else {
        Some(distances[dst])
    };
}

pub fn network_delay_time(data: Vec<Vec<i32>>, nodes: i32, source: i32) -> i32 {
    let nodes = nodes as usize + 1;
    let source = source as usize;
    let mut graph = vec![Vec::new(); nodes];
    for item in data {
        graph[item[0] as usize].push((item[1] as usize, item[2]));
    }
    let mut times = vec![None; nodes];
    times[source] = Some(0);
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((0, source)));

    while let Some(Reverse((time, node))) = heap.pop() {
        for &(adj_node, more_time) in &graph[node] {
            times[adj_node] = match times[adj_node] {
                None => {
                    heap.push(Reverse((time + more_time, adj_node)));
                    Some(time + more_time)
                }
                Some(current_time) => {
                    if time + more_time < current_time {
                        heap.push(Reverse((time + more_time, adj_node)));
                        Some(time + more_time)
                    } else {
                        Some(current_time)
                    }
                }
            };
        }
    }

    times.into_iter().skip(1).try_fold(i32::MIN, |max_val, val| match val {
        None => None,
        Some(num) => Some(max_val.max(num))
    }).unwrap_or(-1)
}

///Multiply start with any number in nums (mod 1e5 if necessary) and find the number of steps to reach the end
pub fn minimum_multiplications_to_reach_end(nums: Vec<i32>, start: i32, end: i32) -> i32 {
    const MOD: i32 = 1e5 as i32;
    let mut multiplications = vec![i32::MAX; MOD as usize]; //Theses are all the numbers from 0..9999
    multiplications[start as usize] = 0;
    //Here a heap is not necessary since the nodes will be traversed in order
    let mut queue = VecDeque::new(); //Steps, number
    queue.push_back((0, start));

    while let Some((step, number)) = queue.pop_front() {
        for &mul in &nums {
            let new_num = (number * mul) % MOD;
            if step + 1 < multiplications[new_num as usize] {
                multiplications[new_num as usize] = step + 1;
                if new_num == end {
                    return step + 1;
                }
                queue.push_back((step + 1, new_num));
            }
        }
    }

    -1
}

//The number of ways to go from 1->n in the shortest time possible
pub fn count_paths(adj_list: Vec<Vec<(usize, i32)>>) -> i32 {
    let mut heap = BinaryHeap::new();
    heap.push(Reverse((0, 0))); //Time, Node
    let mut times = vec![None; adj_list.len()];
    times[0] = Some(0);
    let mut ways = vec![0; adj_list.len()]; //Number of ways node is reached in the shortest time
    ways[0] = 1;

    while let Some(Reverse((time, node))) = heap.pop() {
        for &(adj_node, wait) in &adj_list[node] {
            times[adj_node] = match times[adj_node] {
                None => {
                    heap.push(Reverse((wait + time, adj_node)));
                    ways[adj_node] = ways[node];
                    Some(wait + time)
                }
                Some(current_time) => {
                    match current_time.cmp(&(wait + time)) {
                        Ordering::Less => {
                            Some(current_time)
                        }
                        Ordering::Equal => {
                            ways[adj_node] += ways[node];
                            Some(current_time)
                        }
                        Ordering::Greater => {
                            heap.push(Reverse((wait + time, adj_node)));
                            ways[adj_node] = ways[node];
                            Some(wait + time)
                        }
                    }
                }
            }
        }
    }

    ways[adj_list.len() - 1]
}

///This algorithm also find the shortest path. This can handle negative weights, but it only works for directed graph. T.C. = O(VE)
pub fn bellman_ford_algorithm(adj_list: Vec<Vec<(usize, i32)>>) -> Result<Vec<Option<i32>>, &'static str> {
    //Here, we 'relax' the edges (update the distance) V-1 for each vertex
    let mut distances = vec![None; adj_list.len()];
    distances[0] = Some(0);

    for _ in 0..adj_list.len() - 1 { //N-1 relaxations
        for node in 0..adj_list.len() {
            if let Some(dist) = distances[node] {
                for &(adj_node, weight) in &adj_list[node] {
                    if let Some(adj_dist) = distances[adj_node] {
                        if dist + weight < adj_dist {
                            distances[adj_node] = Some(dist + weight); //Relaxation
                        }
                    } else {
                        distances[adj_node] = Some(dist + weight);
                    }
                }
            }
        }
    }
    //One final relaxation. If the distances can still be decreased, this means there is a negative weight cycle
    for node in 0..adj_list.len() {
        if let Some(dist) = distances[node] {
            for &(adj_node, weight) in &adj_list[node] {
                if let Some(adj_dist) = distances[adj_node] {
                    if dist + weight < adj_dist {
                        return Err("The graph contains a negative cycle");
                    }
                }
            }
        }
    }

    Ok(distances)
}

///This is a multi-source shortest distance algorithm. It can find the distance between every possible pair. T.C. = O(V^3). It is better for dense graphs. A negative cycle is present if distance[i][i] != 0
pub fn floyd_warshall_algorithm(adj_matrix: Vec<Vec<Option<i32>>>) -> Vec<Vec<Option<i32>>> {
    let v = adj_matrix.len();
    let mut distances = adj_matrix.clone();
    for i in 0..v {
        for j in 0..v {
            if i == j {
                distances[i][j] = Some(0); //Distance to self = 0
            }
        }
    }

    for k in 0..v {
        for i in 0..v {
            for j in 0..v {
                distances[i][j] = Some(std::cmp::min(
                    distances[i][j].unwrap_or(i32::MAX), distances[i][k].unwrap_or(i32::MAX) + distances[k][j].unwrap_or(i32::MAX),
                ));
            }
        }
    }

    distances
}

///This is a multi-source shortest path algorithm. It can find the distance between every possible pair. T.C. = O(V^2logV + VE). It is a combination of Dijkstra's and Bellman-Ford Algorithm
///Unlike Floyd-Warshall, it can handle negative weights
pub fn johnson_algorithm(adj_matrix: Vec<Vec<Option<i32>>>) -> Vec<Vec<Option<i32>>> { //This simply converts all edge weights to positive
    //Adding an extra vertex to perform Bellman-Ford. This new node is at a distance 0 from every other node
    let mut added_graph = adj_matrix.clone();
    for row in added_graph.iter_mut() {
        row.push(None);
    }
    added_graph.push(vec![Some(0); added_graph[0].len()]);
    //Apply Bellman-Ford to find the distances from the added vertex
    fn bellman_ford(adj_matrix: Vec<Vec<Option<i32>>>) -> Vec<Option<i32>> {
        let mut distances = vec![None; adj_matrix.len()];
        distances[adj_matrix.len() - 1] = Some(0);

        for _ in 0..adj_matrix.len() - 1 {
            for u in 0..adj_matrix.len() {
                for v in 0..adj_matrix.len() {
                    if let Some(distance_u) = distances[u] {
                        if let Some(edge_weight) = adj_matrix[u][v] {
                            distances[v] = match distances[v] {
                                None => Some(distance_u + edge_weight),
                                Some(distance_v) => {
                                    if distance_u + edge_weight < distance_v {
                                        Some(distance_u + edge_weight)
                                    } else {
                                        Some(distance_v)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        distances
    }

    let mut distances = bellman_ford(added_graph.clone());
    distances.pop(); //Remove the added node

    //Change the edge weights of the graph such that all of them are positive
    //d[v] = d[u] + w(u, v) => w'(u,v) = d[u] - d[v] + w(u, v)
    let mut positive_graph: Vec<Vec<Option<i32>>> = Vec::new();
    added_graph.pop();
    for row in added_graph.into_iter() {
        positive_graph.push(row[..row.len() - 1].iter().cloned().collect());
    }
    for u in 0..positive_graph.len() {
        for v in 0..positive_graph.len() {
            if let Some(mut weight) = positive_graph[u][v] {
                if let (Some(du), Some(dv)) = (distances[u], distances[v]) {
                    positive_graph[u][v] = Some(weight + du - dv);
                }
            }
        }
    }
    //Now all the weights have been made positive
    positive_graph
}

pub fn find_the_city(cities: i32, edges: Vec<Vec<i32>>, distance_threshold: i32) -> Option<usize> {
    let cities = cities as usize;
    let mut distances = vec![vec![None; cities]; cities];
    for edge in edges {
        distances[edge[0] as usize][edge[1] as usize] = Some(edge[2]);
        distances[edge[1] as usize][edge[0] as usize] = Some(edge[2]);
    }

    for i in 0..cities {
        distances[i][i] = Some(0);
    }

    for via in 0..cities {
        for i in 0..cities {
            for j in 0..cities {
                match (distances[i][via], distances[via][j]) {
                    (Some(dist_via_i), Some(dist_via_j)) => distances[i][j] = Some(std::cmp::min(
                        distances[i][j].unwrap_or(i32::MAX), dist_via_i + dist_via_j,
                    )),
                    (Some(_), None) => continue,
                    (None, Some(_)) => continue,
                    (None, None) => continue
                }
            }
        }
    }

    distances.iter()
        .map(|row| row.iter()
            .filter_map(|&x| x)
            .filter(|&x| x <= distance_threshold)
            .count() - 1)
        .enumerate()
        .min_by_key(|&(city, count)| (count, Reverse(city)))
        .map(|(city, _)| city)
}

/*
 A spanning tree is a subset of a graph G, such that all the vertices are connected using the minimum possible number of edges, i.e. if there are n nodes and n-1 edges and all nodes are reachable from each other.
 Thus, a spanning tree has no cycles and a graph may have more than one spanning tree
 A minimum spanning tree is the spanning tree with minimum weight (sum of all weights) among all the possible spanning trees
*/
///It is a greedy algorithm to build the MST of a graph. T.C. = O(V^2) (adjacency matrix) O((V+E) logV) (adjacency list)
pub fn prim_algorithm(adj_list: Vec<Vec<(usize, i32)>>) -> (i32, Vec<(usize, usize)>) {
    let mut vis = vec![false; adj_list.len()];
    let mut heap = BinaryHeap::new(); //Weight, Node, Parent
    heap.push(Reverse((0, (0, None))));
    let mut weight_sum = 0;
    let mut mst_edges = Vec::new();

    while let Some(Reverse((weight, (node, parent)))) = heap.pop() {
        if !vis[node] {
            vis[node] = true;
            weight_sum += weight;
            if let Some(parent) = parent {
                mst_edges.push((parent, node));
            }

            for &(adj_node, edge_weight) in &adj_list[node] {
                if !vis[adj_node] {
                    heap.push(Reverse((edge_weight, (adj_node, Some(node)))));
                }
            }
        }
    }

    (weight_sum, mst_edges)
}

// A Disjoint Set (DSU) or Union-Find Structures are generally used in problems with connections and dividing things in groups.
pub struct DisjointSet {
    parent: Vec<usize>, //This stores the super-parent (parent of parent...)
    rank: Vec<usize>, //This generally refers to the distance between the node and the furthest leaf node
    size: Vec<usize>, //This is the number of nodes below any particular node
}

//In a problem, we can use either Union by Rank or by Size but not both
impl DisjointSet {
    pub fn new(n: usize) -> Self {
        let parent: Vec<usize> = (0..n).collect();
        let rank = vec![0; n];
        let size = vec![1; n];

        DisjointSet { parent, rank, size }
    }

    pub fn find_ultimate_parent(&mut self, node: usize) -> usize { //T.C. = O(log n)
        if self.parent[node] == node {
            node
        } else {
            self.parent[node] = self.find_ultimate_parent(self.parent[node]); //This is the path compression step
            self.parent[node]
        }
    }
    ///First find the ranks of the ultimate parents of u and v. Connect the smaller rank to the larger rank
    pub fn union_by_rank(&mut self, u: usize, v: usize) {
        let ultimate_parent_u = self.find_ultimate_parent(u);
        let ultimate_parent_v = self.find_ultimate_parent(v);

        if ultimate_parent_u == ultimate_parent_v { //If they are already in the same component
            return; //There is no point in connecting them
        }

        match self.rank[ultimate_parent_u].cmp(&self.rank[ultimate_parent_v]) {
            Ordering::Less => self.parent[ultimate_parent_u] = ultimate_parent_v,
            Ordering::Equal => self.parent[ultimate_parent_v] = ultimate_parent_u,
            Ordering::Greater => {
                self.parent[ultimate_parent_u] = ultimate_parent_v; //Since they are equal we can do the other way as well
                self.rank[ultimate_parent_u] += 1;
            }
        }
    }

    pub fn union_by_size(&mut self, u: usize, v: usize) {
        let ultimate_parent_u = self.find_ultimate_parent(u);
        let ultimate_parent_v = self.find_ultimate_parent(v);

        if ultimate_parent_u == ultimate_parent_v {
            return;
        }

        match self.size[ultimate_parent_u].cmp(&self.size[ultimate_parent_v]) {
            Ordering::Less => {
                self.parent[ultimate_parent_u] = ultimate_parent_v;
                self.size[ultimate_parent_v] += self.size[ultimate_parent_u];
            }
            Ordering::Equal => {
                self.parent[ultimate_parent_v] = ultimate_parent_u;
                self.size[ultimate_parent_u] += self.size[ultimate_parent_v];
            }
            Ordering::Greater => {
                self.parent[ultimate_parent_u] = ultimate_parent_v;
                self.size[ultimate_parent_v] += self.size[ultimate_parent_u];
            }
        }
    }
}

///It is also used to find the MST. T.C. = O()
pub fn kruskal_algorithm(adj_list: Vec<Vec<(usize, i32)>>) -> (i32, Vec<(usize, usize)>) {
    let mut edges = Vec::new();

    for vertex in 0..adj_list.len() {
        for &(adj_node, weight) in &adj_list[vertex] {
            //Since the graph is bidirectional, each edge will be stored twice but the Disjoint Set will take care of it
            edges.push((weight, vertex, adj_node));
        }
    }

    edges.sort();

    let mut ds = DisjointSet::new(adj_list.len());
    let mut weight_sum = 0;
    let mut mst_edges = Vec::new();

    for &(weight, u, v) in &edges {
        if ds.find_ultimate_parent(u) != ds.find_ultimate_parent(v) {
            weight_sum += weight;
            mst_edges.push((u, v));
            ds.union_by_rank(u, v);
        }
    }

    (weight_sum, mst_edges)
}

pub fn number_of_province_dsu(adj_matrix: Vec<Vec<i32>>) -> i32 {
    //All connected components will have the same ultimate parent
    let mut ds = DisjointSet::new(adj_matrix.len());
    for i in 0..adj_matrix.len() {
        for j in 0..adj_matrix.len() {
            if adj_matrix[i][j] == 1 {
                ds.union_by_rank(i, j);
            }
        }
    }

    ds.parent.iter().enumerate()
        .filter(|&(index, &parent)| index == parent).count() as i32
}

pub fn make_connected(computers: i32, connections: Vec<Vec<i32>>) -> i32 {
    let mut ds = DisjointSet::new(computers as usize);
    let mut extra_cables = 0; //These extra edges can be used to connect non-connected components
    for edge in connections {
        let (u, v) = (edge[0] as usize, edge[1] as usize);
        if ds.find_ultimate_parent(u) == ds.find_ultimate_parent(v) {
            extra_cables += 1;
        } else {
            ds.union_by_rank(u, v);
        }
    }
    let components = ds.parent.iter().enumerate() //These are the non-connected components
        .filter(|&(index, &parent)| index == parent).count() as i32;

    let required_cables = components - 1; //Cables that will be required to connect the components

    if extra_cables >= required_cables {
        required_cables
    } else {
        -1
    }
}

///If two entries with the same name have one same email, then merge in the sorted order.
pub fn accounts_merge(accounts: Vec<Vec<String>>) -> Vec<Vec<String>> {
    let mut mail_map = HashMap::new(); //Mail, Parent of the mail (index)
    let mut ds = DisjointSet::new(accounts.len());
    //Create the Disjoint Set
    for (index, &ref account) in accounts.iter().enumerate() {
        for mail in account.iter().skip(1) {
            if let Some(&prev) = mail_map.get(mail) {
                ds.union_by_size(index, prev); //If mail already in the account, make old index the parent
            } else {
                mail_map.insert(mail.clone(), index); //Add to map
            }
        }
    }

    let mut merged_mails = vec![Vec::new(); accounts.len()];

    for (mail, parent) in mail_map {
        let index = ds.find_ultimate_parent(parent);
        merged_mails[index].push(mail);
    }

    merged_mails.iter_mut().for_each(|x| x.sort()); //Sort the accounts

    for (i, merged_mail) in merged_mails.iter_mut().enumerate() { //Add names to the accounts
        if merged_mail.is_empty() {
            continue;
        }

        merged_mail.insert(0, accounts[i][0].clone());
    }

    merged_mails.retain(|account| !account.is_empty()); //Remove empty accounts

    merged_mails
}

//This is the best problem for a DSU since the graph is changing (node queries[i] are added).
///We need to find the number of islands after each query
pub fn num_islands_2(rows: usize, columns: usize, queries: Vec<(usize, usize)>) -> Vec<i32> {
    let mut vis = vec![vec![false; columns]; rows];
    const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let cell_number = |row: usize, col: usize| { row * columns + col }; //Give each cell a corresponding number
    let is_valid = |row: i32, col: i32| { row >= 0 && row < rows as i32 && col >= 0 && col < columns as i32 };
    let mut ds = DisjointSet::new(rows * columns);
    let mut islands = 0;
    let mut responses = Vec::with_capacity(queries.len());

    for (row, col) in queries {
        if vis[row][col] {
            responses.push(islands);
            continue;
        }

        vis[row][col] = true;
        islands += 1;

        for direction in DIRECTIONS {
            let new_row = row as i32 + direction.0;
            let new_col = col as i32 + direction.1;

            if is_valid(new_row, new_col) {
                let new_row = new_row as usize;
                let new_col = new_col as usize;
                if vis[new_row][new_col] {
                    let cell = cell_number(row, col);
                    let adj_cell = cell_number(new_row, new_col);

                    if ds.find_ultimate_parent(cell) != ds.find_ultimate_parent(adj_cell) { //If the adjacent cells connect,
                        islands -= 1; // reduce 1 island
                        ds.union_by_size(cell, adj_cell);
                    }
                }
            }
        }

        responses.push(islands);
    }

    responses
}

///Change at most one cell from 0 -> 1 and return the size of longest island
pub fn largest_island(grid: Vec<Vec<i32>>) -> i32 {
    let n = grid.len();
    const DIRECTIONS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let cell_number = |row: usize, col: usize| { row * n + col }; //Give each cell a corresponding number
    let is_valid = |row: i32, col: i32| { row >= 0 && row < n as i32 && col >= 0 && col < n as i32 };
    let mut ds = DisjointSet::new(n * n);

    for row in 0..n {
        for col in 0..n {
            if grid[row][col] == 0 {
                continue;
            }

            for direction in DIRECTIONS {
                let new_row = row as i32 + direction.0;
                let new_col = col as i32 + direction.1;

                if is_valid(new_row, new_col) {
                    let new_row = new_row as usize;
                    let new_col = new_col as usize;
                    let cell = cell_number(row, col);
                    let adj_cell = cell_number(new_row, new_col);
                    if grid[new_row][new_col] == 1 {
                        ds.union_by_size(cell, adj_cell);
                    }
                }
            }
        }
    }

    let mut max_size = 0;
    for row in 0..n {
        for col in 0..n {
            if grid[row][col] == 1 {
                continue;
            }
            let mut set = HashSet::new();
            for direction in DIRECTIONS {
                let new_row = row as i32 + direction.0;
                let new_col = col as i32 + direction.1;

                if is_valid(new_row, new_col) {
                    let new_row = new_row as usize;
                    let new_col = new_col as usize;
                    let adj_cell = cell_number(new_row, new_col);
                    if grid[new_row][new_col] == 1 {
                        set.insert(ds.find_ultimate_parent(adj_cell));
                    }
                }
            }
            max_size = max_size.max(1 + set.iter().map(|&x| ds.size[x]).sum::<usize>());
        }
    }

    for cell in 0..n * n {
        let ult = ds.find_ultimate_parent(cell);
        max_size = max_size.max(ds.size[ult]);
    }

    max_size as i32
}

///A stone can be removed if it shares the same row/column with a stone that has not been removed
pub fn remove_stones(stones: Vec<Vec<i32>>) -> i32 {
    //Consider stones in the same row or column to form a component. We can remove all but 1 from each of these components.
    //Consider k components each with n_i stones. The stones which can be removed are = (n_1-1)+(n_2-1)+... = n-k
    let (mut rows, mut columns) = (0, 0);
    for pos in &stones {
        rows = rows.max(pos[0] as usize);
        columns = columns.max(pos[1] as usize);
    }
    //Change the column numbers to get indices different from rows
    let mut ds = DisjointSet::new(rows + columns + 2);
    let mut stone_nodes = HashSet::new();

    for pos in &stones {
        let row = pos[0] as usize;
        let col = pos[1] as usize + rows + 1;
        ds.union_by_size(row, col);
        stone_nodes.insert(row);
        stone_nodes.insert(col);
    }

    let mut components = 0;
    for node in stone_nodes {
        if ds.find_ultimate_parent(node) == node {
            components += 1;
        }
    }

    stones.len() as i32 - components
}

//A graph is said to be strongly connected if every vertex is reachable from every other vertex
///Kosaraju's Algorithm is used to find the strongly connected components of a directed graph. T.C. = O(V+E) for adjacency list and O(V^2) for adjacency matrix
pub fn kosaraju_algorithm(adj_list: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut vis = vec![false; adj_list.len()];
    let mut stack = VecDeque::new();
    fn helper(node: usize, adj_list: &Vec<Vec<usize>>, vis: &mut Vec<bool>, stack: &mut VecDeque<usize>) { //DFS
        vis[node] = true;
        for &adj_node in &adj_list[node] {
            if !vis[adj_node] {
                helper(adj_node, adj_list, vis, stack);
            }
        }
        stack.push_back(node);
    }

    for node in 0..adj_list.len() {
        if !vis[node] {
            helper(node, &adj_list, &mut vis, &mut stack);
        }
    }

    //Reverse the graph
    let mut reversed_adj_list = vec![Vec::new(); adj_list.len()];
    for node in 0..adj_list.len() {
        vis[node] = false; //Unmark the node
        for &adj_node in &adj_list[node] {
            reversed_adj_list[adj_node].push(node);
        }
    }

    let mut scc = Vec::new();
    let mut component = Vec::new();
    fn helper_rev(node: usize, adj_list: &Vec<Vec<usize>>, vis: &mut Vec<bool>, component: &mut Vec<usize>) {
        vis[node] = true;
        component.push(node);

        for &adj_node in &adj_list[node] {
            if !vis[adj_node] {
                helper_rev(adj_node, adj_list, vis, component);
            }
        }
    }

    while let Some(node) = stack.pop_back() {
        if !vis[node] {
            helper_rev(node, &reversed_adj_list, &mut vis, &mut component);
            component.sort();
            scc.push(component.clone());
            component.clear();
        }
    }

    scc
}

//A bridge is an edge which when broken results in 2 or more components
pub fn tarjan_algorithm(adj_list: Vec<Vec<usize>>) -> Vec<(usize, usize)> {
    let mut vis = vec![false; adj_list.len()];
    let mut insertion_time = vec![0; adj_list.len()]; //This is the time at which the dfs call reaches the node
    let mut lowest_visit_time = vec![0; adj_list.len()]; //This is the lowest time of all adjacent nodes (except parent)
    let mut bridges = Vec::new();

    fn helper(node: usize, parent: Option<usize>, adj_list: &Vec<Vec<usize>>, vis: &mut Vec<bool>, mut timer: i32, insertion_time: &mut Vec<i32>, lowest_visit_time: &mut Vec<i32>, bridges: &mut Vec<(usize, usize)>) { //DFS Traversal
        vis[node] = true;

        insertion_time[node] = timer;
        lowest_visit_time[node] = timer;
        timer += 1;

        for &adj_node in &adj_list[node] {
            if Some(adj_node) != parent {
                if !vis[adj_node] {
                    helper(adj_node, Some(node), adj_list, vis, timer, insertion_time, lowest_visit_time, bridges);
                    lowest_visit_time[node] = lowest_visit_time[node].min(lowest_visit_time[adj_node]);

                    if lowest_visit_time[adj_node] > insertion_time[node] { //If it is a bridge
                        bridges.push((node, adj_node));
                    }
                } else { //This cannot be a bridge
                    lowest_visit_time[node] = lowest_visit_time[node].min(lowest_visit_time[adj_node]);
                }
            }
        }
    }

    helper(0, None, &adj_list, &mut vis, 1, &mut insertion_time, &mut lowest_visit_time, &mut bridges);

    bridges
}

///An articulation point is a node on whose removal breaks the graph into multiple components
pub fn articulation_points(adj_list: Vec<Vec<usize>>) -> Vec<usize> {
    let mut vis = vec![false; adj_list.len()];
    let mut insertion_time = vec![0; adj_list.len()];
    let mut lowest_visit_time = vec![0; adj_list.len()]; //This is the lowest time of all adjacent nodes (except parent and visited nodes)
    let mut points = Vec::new();

    fn helper(node: usize, parent: Option<usize>, adj_list: &Vec<Vec<usize>>, vis: &mut Vec<bool>, mut timer: i32, insertion_time: &mut Vec<i32>, lowest_visit_time: &mut Vec<i32>, points: &mut Vec<usize>) {
        vis[node] = true;

        insertion_time[node] = timer;
        lowest_visit_time[node] = timer;
        timer += 1;

        for &adj_node in &adj_list[node] {
            if Some(adj_node) == parent {
                continue;
            }

            if !vis[adj_node] {
                helper(adj_node, Some(node), adj_list, vis, timer, insertion_time, lowest_visit_time, points);
                lowest_visit_time[node] = lowest_visit_time[node].min(lowest_visit_time[adj_node]);

                if lowest_visit_time[adj_node] >= insertion_time[node] && parent.is_some() {
                    points.push(node);
                }
            } else {
                lowest_visit_time[node] = lowest_visit_time[node].min(lowest_visit_time[adj_node]);
            }
        }
    }

    helper(0, None, &adj_list, &mut vis, 0, &mut insertion_time, &mut lowest_visit_time, &mut points);

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfs_test() {
        let graph = vec![
            vec![],
            vec![2, 3],
            vec![1, 3, 4],
            vec![1, 2, 4],
            vec![2, 3],
        ];
        assert_eq!(breadth_first_search(graph), vec![1, 2, 3, 4]);

        let graph = vec![
            vec![],
            vec![2, 6],
            vec![1, 3, 4],
            vec![2],
            vec![2, 5],
            vec![4, 8],
            vec![1, 7, 9],
            vec![6, 8],
            vec![5, 7],
            vec![6],
        ];
        assert_eq!(breadth_first_search(graph), vec![1, 2, 6, 3, 4, 7, 9, 5, 8]);
    }

    #[test]
    fn dfs_test() {
        let graph = vec![
            vec![],
            vec![2, 3],
            vec![1, 5, 6],
            vec![1, 4, 7],
            vec![3, 8],
            vec![2],
            vec![2],
            vec![3, 8],
            vec![4, 7],
        ];
        assert_eq!(depth_first_search_recursive(graph.clone()), vec![1, 2, 5, 6, 3, 4, 8, 7]);
        assert_eq!(depth_first_search_iterative(graph), vec![1, 2, 5, 6, 3, 4, 8, 7]);
    }

    #[test]
    fn provinces_test() {
        assert_eq!(number_of_provinces(vec![
            vec![1, 1, 0],
            vec![1, 1, 0],
            vec![0, 0, 1],
        ]), 2);
        assert_eq!(number_of_provinces(vec![
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
        ]), 3);
    }

    #[test]
    fn flood_fill_test() {
        assert_eq!(flood_fill(vec![
            vec![1, 1, 1],
            vec![1, 1, 0],
            vec![1, 0, 1],
        ], 1, 1, 2), vec![
            vec![2, 2, 2],
            vec![2, 2, 0],
            vec![2, 0, 1],
        ]);
    }

    #[test]
    fn orange_test() {
        assert_eq!(oranges_rotting(vec![
            vec![2, 1, 1],
            vec![1, 1, 0],
            vec![0, 1, 1],
        ]), Some(4));
        assert_eq!(oranges_rotting(vec![
            vec![2, 1, 1],
            vec![0, 1, 1],
            vec![1, 0, 1],
        ]), None);
    }

    #[test]
    fn cycle_detection_test() {
        let graph = vec![
            vec![],
            vec![1],
            vec![0, 2, 4],
            vec![1, 3],
            vec![2, 4],
            vec![1, 3],
        ];
        assert_eq!(cycle_detection_bfs(graph.clone()), true);
        assert_eq!(cycle_detection_dfs(graph), true);

        let graph = vec![
            vec![],
            vec![2],
            vec![1, 3],
            vec![2],
        ];
        assert_eq!(cycle_detection_bfs(graph.clone()), false);
        assert_eq!(cycle_detection_dfs(graph), false);
    }

    #[test]
    fn distance_test() {
        assert_eq!(update_matrix(vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 0],
        ]), vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 0],
        ]);
        assert_eq!(update_matrix(vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![1, 1, 1],
        ]), vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![1, 2, 1],
        ]);
    }

    #[test]
    fn surround_regions_test() {
        let mut board = vec![
            vec!['X', 'X', 'X', 'X'],
            vec!['X', 'O', 'O', 'X'],
            vec!['X', 'X', 'O', 'X'],
            vec!['X', 'O', 'X', 'X'],
        ];
        surround_regions(&mut board);
        assert_eq!(board, vec![
            vec!['X', 'X', 'X', 'X'],
            vec!['X', 'X', 'X', 'X'],
            vec!['X', 'X', 'X', 'X'],
            vec!['X', 'O', 'X', 'X'],
        ]);
    }

    #[test]
    fn enclaves_test() {
        assert_eq!(num_enclaves(vec![
            vec![0, 0, 0, 0],
            vec![1, 0, 1, 0],
            vec![0, 1, 1, 0],
            vec![0, 0, 0, 0],
        ]), 3);
        assert_eq!(num_enclaves(vec![
            vec![0, 1, 1, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 0],
        ]), 0);
    }

    #[test]
    fn is_bipartite_test() {
        assert_eq!(is_bipartite_bfs(vec![
            vec![1, 2, 3],
            vec![0, 2],
            vec![0, 1, 3],
            vec![0, 2],
        ]), false);
        assert_eq!(is_bipartite_dfs(vec![
            vec![1, 2, 3],
            vec![0, 2],
            vec![0, 1, 3],
            vec![0, 2],
        ]), false);
        assert_eq!(is_bipartite_bfs(vec![
            vec![1, 3],
            vec![0, 2],
            vec![1, 3],
            vec![0, 2],
        ]), true);
        assert_eq!(is_bipartite_dfs(vec![
            vec![1, 3],
            vec![0, 2],
            vec![1, 3],
            vec![0, 2],
        ]), true);
    }

    #[test]
    fn eventual_safe_nodes_test() {
        assert_eq!(eventual_safe_nodes(vec![
            vec![1, 2],
            vec![2, 3],
            vec![5],
            vec![0],
            vec![5],
            vec![],
            vec![],
        ]), vec![2, 4, 5, 6]);
        assert_eq!(eventual_safe_nodes(vec![
            vec![1, 2, 3, 4],
            vec![1, 2],
            vec![3, 4],
            vec![0, 4],
            vec![],
        ]), vec![4]);
    }

    #[test]
    fn topological_sort_test() {
        let graph = vec![
            vec![],
            vec![2, 4],
            vec![3],
            vec![6],
            vec![5],
            vec![3],
            vec![],
        ];
        assert_eq!(topological_sort(graph.clone()), vec![1, 4, 5, 2, 3, 6]);
        let graph = vec![
            vec![],
            vec![],
            vec![3],
            vec![1],
            vec![0, 1],
            vec![0, 2],
        ];
        assert_eq!(kahn_algorithm(graph), vec![4, 5, 0, 2, 3, 1]);
    }

    #[test]
    fn cycle_detection_with_topo_test() {
        let graph = vec![
            vec![],
            vec![],
            vec![3],
            vec![1],
            vec![0, 1],
            vec![0, 2],
        ];
        assert_eq!(cycle_directed_topological(graph), false);
        let graph = vec![
            vec![],
            vec![2],
            vec![3],
            vec![4, 5],
            vec![2],
            vec![],
        ];
        assert_eq!(cycle_directed_topological(graph), true);
    }

    #[test]
    fn course_schedule_1_test() {
        assert_eq!(can_finish(2, vec![(1, 0)]), true);
        assert_eq!(can_finish(2, vec![(1, 0), (0, 1)]), false);
    }

    #[test]
    fn eventual_safe_nodes_bfs_test() {
        assert_eq!(eventual_safe_nodes_bfs(vec![
            vec![1, 2],
            vec![2, 3],
            vec![5],
            vec![0],
            vec![5],
            vec![],
            vec![],
        ]), vec![2, 4, 5, 6]);
        assert_eq!(eventual_safe_nodes_bfs(vec![
            vec![1, 2, 3, 4],
            vec![1, 2],
            vec![3, 4],
            vec![0, 4],
            vec![],
        ]), vec![4]);
    }

    #[test]
    fn alien_dictionary_test() {
        let dict = vec!["baa", "abcd", "abca", "cab", "cad"]
            .iter()
            .map(|&s| s.to_string())
            .collect();
        assert_eq!(alien_dictionary(dict, 4), "bdac");

        let dict = vec!["caa", "aaa", "aab"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(alien_dictionary(dict, 3), "cab");
    }

    #[test]
    fn shortest_path_in_dag_test() {
        let graph: Vec<Vec<(usize, i32)>> = vec![
            vec![(1, 2), (4, 1)],
            vec![(2, 3)],
            vec![(3, 6)],
            vec![],
            vec![(2, 2), (5, 4)],
            vec![(3, 1)],
        ];
        assert_eq!(shortest_path_in_dag(graph), vec![Some(0), Some(2), Some(3), Some(6), Some(1), Some(5)]);
    }

    #[test]
    fn word_ladder_test() {
        assert_eq!(ladder_length("hit".to_string(), "cog".to_string(), vec!["hot", "dot", "dog", "lot", "log", "cog"].iter().map(|x| x.to_string()).collect()), 5);
    }

    #[test]
    fn find_ladder_test() {
        assert_eq!(
            find_ladder_optimised("hit".to_string(), "cog".to_string(), vec!["hot", "dot", "dog", "lot", "log", "cog"].iter().map(|x| x.to_string()).collect()),
            vec![
                vec!["hit", "hot", "dot", "dog", "cog"].iter().map(|&s| s.to_string()).collect::<Vec<String>>(),
                vec!["hit", "hot", "lot", "log", "cog"].iter().map(|&s| s.to_string()).collect(),
            ]
        );
        assert_eq!(
            find_ladder_optimised("a".to_string(), "c".to_string(), vec!["a", "b", "c"].iter().map(|x| x.to_string()).collect()),
            vec![
                vec!["a", "c"].iter().map(|x| x.to_string()).collect::<Vec<String>>()
            ]
        );
    }

    #[test]
    fn dijkstra_algorithm_test() {
        let data = vec![
            vec![(1, 1), (2, 6)],
            vec![(2, 3), (0, 1)],
            vec![(1, 3), (0, 6)],
        ];
        assert_eq!(dijkstra_algorithm_min_heap(data.clone(), 2), vec![Some(4), Some(3), Some(0)]);
    }

    #[test]
    fn shortest_path_test() {
        let graph = vec![
            vec![],
            vec![(2, 2), (4, 1)],
            vec![(1, 2), (3, 4), (5, 5)],
            vec![(2, 4), (4, 3), (5, 1)],
            vec![(1, 1), (3, 3)],
            vec![(2, 5), (3, 3)],
        ];
        assert_eq!(shortest_path(graph), Some(vec![1, 4, 3, 5]));
    }

    #[test]
    fn binary_maze_shortest_test() {
        let graph: Vec<Vec<i32>> = vec![
            vec![0, 0, 0],
            vec![1, 1, 0],
            vec![1, 1, 0],
        ];
        assert_eq!(shortest_path_binary_matrix(graph), 4);
    }

    #[test]
    fn minimum_effort_test() {
        let matrix: Vec<Vec<i32>> = vec![
            vec![1, 2, 2],
            vec![3, 8, 2],
            vec![5, 3, 5],
        ];
        assert_eq!(minimum_effort_path(matrix), 2);
        let matrix: Vec<Vec<i32>> = vec![
            vec![1, 2, 3],
            vec![3, 8, 4],
            vec![5, 3, 5],
        ];
        assert_eq!(minimum_effort_path(matrix), 1);
    }

    #[test]
    fn cheapest_flight_test() {
        let graph = vec![
            vec![(1, 100)],
            vec![(2, 100), (3, 600)],
            vec![(0, 100)],
            vec![],
        ];
        assert_eq!(find_cheapest_price(4, graph, 0, 3, 1), Some(700));
    }

    #[test]
    fn network_test() {
        assert_eq!(network_delay_time(vec![vec![1, 2, 1]], 2, 2), -1);
    }

    #[test]
    fn min_multiplications_test() {
        assert_eq!(minimum_multiplications_to_reach_end(vec![2, 5, 7], 3, 30), 2);
        assert_eq!(minimum_multiplications_to_reach_end(vec![3, 4, 65], 7, 66175), 4);
    }

    #[test]
    fn count_paths_test() {
        let data: Vec<Vec<(usize, i32)>> = vec![
            vec![(6, 7), (1, 2), (4, 5)],
            vec![(0, 2), (2, 3), (3, 3)],
            vec![(1, 3), (5, 1)],
            vec![(1, 3), (6, 3), (5, 1)],
            vec![(0, 5), (6, 2)],
            vec![(3, 1), (6, 1), (2, 1)],
            vec![(0, 7), (3, 3), (5, 1), (4, 2)],
        ];
        assert_eq!(count_paths(data), 4);
    }

    #[test]
    fn bellman_ford_test() {
        let data = vec![
            vec![(1, 1), (2, 6)],
            vec![(2, 3), (0, 1)],
            vec![(1, 3), (0, 6)],
        ];
        assert_eq!(bellman_ford_algorithm(data), Ok(vec![Some(0), Some(1), Some(4)]));
    }

    #[test]
    fn find_city_test() {
        assert_eq!(find_the_city(4, vec![
            vec![0, 1, 3],
            vec![1, 2, 1],
            vec![1, 3, 4],
            vec![2, 3, 1],
        ], 4), Some(3));

        assert_eq!(find_the_city(5, vec![
            vec![0, 1, 2],
            vec![0, 4, 8],
            vec![1, 2, 3],
            vec![1, 4, 2],
            vec![2, 3, 1],
            vec![3, 4, 1],
        ], 2), Some(0));
    }

    #[test]
    fn johnson_algorithm_test() {
        let graph = vec![
            vec![Some(0), Some(4), None, None, Some(1)],
            vec![None, Some(0), None, None, None],
            vec![None, Some(7), Some(0), Some(-2), None],
            vec![None, Some(1), None, Some(0), None],
            vec![None, None, None, Some(-5), Some(0)],
        ];

        let positive = vec![
            vec![Some(0), Some(8), None, None, Some(1)],
            vec![None, Some(0), None, None, None],
            vec![None, Some(11), Some(0), Some(3), None],
            vec![None, Some(0), None, Some(0), None],
            vec![None, None, None, Some(0), Some(0)],
        ];

        assert_eq!(johnson_algorithm(graph), positive);
    }

    #[test]
    fn prim_algorithm_test() {
        let graph = vec![
            vec![(1, 2), (2, 1)],
            vec![(0, 2), (2, 1)],
            vec![(0, 1), (1, 1), (3, 2), (4, 2)],
            vec![(2, 2), (4, 1)],
            vec![(2, 2), (3, 2)],
        ];

        assert_eq!(prim_algorithm(graph), (5, vec![(0, 2), (2, 1), (2, 3), (3, 4)]));
    }

    #[test]
    fn kruskal_algorithm_test() {
        let graph = vec![
            vec![(1, 2), (2, 1)],
            vec![(0, 2), (2, 1)],
            vec![(0, 1), (1, 1), (3, 2), (4, 2)],
            vec![(2, 2), (4, 1)],
            vec![(2, 2), (3, 2)],
        ];

        assert_eq!(kruskal_algorithm(graph), (5, vec![(0, 2), (1, 2), (3, 4), (2, 3)]));
    }

    #[test]
    fn provinces_test_with_dsu() {
        assert_eq!(number_of_province_dsu(vec![
            vec![1, 1, 0],
            vec![1, 1, 0],
            vec![0, 0, 1],
        ]), 2);
        assert_eq!(number_of_province_dsu(vec![
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
        ]), 3);
    }

    #[test]
    fn make_connected_test() {
        assert_eq!(make_connected(4, vec![vec![0, 1], vec![0, 2], vec![1, 2]]), 1);
        assert_eq!(make_connected(11, vec![
            vec![1, 4],
            vec![0, 3],
            vec![1, 3],
            vec![3, 7],
            vec![2, 7],
            vec![0, 1],
            vec![2, 4],
            vec![3, 6],
            vec![5, 6],
            vec![6, 7],
            vec![4, 7],
            vec![0, 7],
            vec![5, 7],
        ]), 3);
    }

    #[test]
    fn merge_accounts_test() {
        let accounts = vec![
            vec!["John".to_string(), "johnsmith@mail.com".to_string(), "john_newyork@mail.com".to_string()],
            vec!["John".to_string(), "johnsmith@mail.com".to_string(), "john00@mail.com".to_string()],
            vec!["Mary".to_string(), "mary@mail.com".to_string()],
            vec!["John".to_string(), "johnnybravo@mail.com".to_string()],
        ];
        accounts_merge(accounts);
    }

    #[test]
    fn number_of_islands_two_test() {
        assert_eq!(num_islands_2(3, 3, vec![(0, 0), (0, 1), (1, 2), (2, 1)]), vec![1, 1, 2, 3]);
    }

    #[test]
    fn large_island_test() {
        assert_eq!(largest_island(vec![vec![1, 0], vec![0, 1]]), 3);
        assert_eq!(largest_island(vec![vec![1, 1], vec![1, 0]]), 4);
        assert_eq!(largest_island(vec![vec![1, 1], vec![1, 1]]), 4);
    }

    #[test]
    fn stone_removal_test() {
        assert_eq!(remove_stones(vec![
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 2],
            vec![2, 1],
            vec![2, 2],
        ]), 5);

        assert_eq!(remove_stones(vec![
            vec![0, 0],
            vec![0, 2],
            vec![1, 1],
            vec![2, 0],
            vec![2, 2],
        ]), 3);

        assert_eq!(remove_stones(vec![
            vec![4, 5],
            vec![0, 4],
            vec![0, 5],
            vec![4, 3],
            vec![2, 2],
            vec![5, 1],
            vec![0, 3],
            vec![2, 4],
            vec![4, 0],
        ]), 7);
    }

    #[test]
    fn kosaraju_test() {
        let mut graph = vec![
            vec![1],
            vec![2],
            vec![0, 3],
            vec![4],
            vec![5, 7],
            vec![6],
            vec![4],
            vec![],
        ];

        assert_eq!(kosaraju_algorithm(graph), vec![vec![0, 1, 2], vec![3], vec![4, 5, 6], vec![7]]);
    }

    #[test]
    fn bridges_test() {
        let graph = vec![
            vec![1, 2],
            vec![0, 2, 3],
            vec![0, 1],
            vec![1],
        ];

        assert_eq!(tarjan_algorithm(graph), vec![(1, 3)]);
    }
}
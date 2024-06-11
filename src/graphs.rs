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

use std::collections::VecDeque;

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

pub fn number_of_distinct_islands(grid: Vec<Vec<i32>>) {
    todo!()
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

///Replace any region (of Os) surrounded by Xs on all 4 sides with X
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

pub fn num_enclaves(grid: Vec<Vec<i32>>) -> i32 {
    todo!()
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
}
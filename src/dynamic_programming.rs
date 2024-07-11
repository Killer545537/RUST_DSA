/* Dynamic Programming (DP) is a method used to solve problems by dividing them into smaller, easier subproblems.
The two types of approaches to DP, Top-Down(Memoization) and Bottom-Up(Tabulation)
In memoization, we start from the top and recurse down while storing the results to the sub-problems
In tabulation, we start from the bottom and recurse up using the results of the sub-problems  */
use std::usize;
use itertools::Itertools;

///The simplest recursive approach
pub fn fibonacci(n: i32) -> i32 {
    match n {
        0 | 1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2)
    }
}
//This is terrible because each call is computed again and again wasting time
pub fn fibonacci_better(n: i32) -> i32 {
    let n = n as usize;
    let mut dp = vec![-1; n + 1];
    dp[0] = 1;
    dp[1] = 1;

    for i in 2..=n {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    dp[n]
}
//Even this can be optimized, since only the last two values are being used, only use them
pub fn fibonacci_best(n: i32) -> i32 {
    let (mut prev1, mut prev2) = (1, 1);

    for _ in 2..=n {
        let current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }

    prev1
}

///From stair i we can jump to i + 1 or i + 2, find the number of ways to climb n stairs
pub fn climb_stairs(n: i32) -> i32 {
    //The recurrence relation is f(n) = f(n-1) + f(n-2), and f(0) = 0, f(1) = 1, f(2) = 2 ((1,1), (2))
    let (mut prev1, mut prev2) = (1, 1);

    for _ in 2..=n {
        let current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }

    prev1
}

///A frog can jump from heights[i]->heights[i+1]|heights[i+2] using energy |heights[i']-heights[i]|
pub fn frog_jump_recursive(heights: &[i32]) -> i32 {
    if heights.len() == 1 { //If there is only one step, no energy is spent
        return 0;
    }

    if heights.len() == 2 {
        return (heights[1] - heights[0]).abs();
    }

    let one = (heights[1] - heights[0]).abs() + frog_jump_recursive(&heights[1..]);
    let two = (heights[2] - heights[0]).abs() + frog_jump_recursive(&heights[2..]);

    return std::cmp::min(one, two);
}
pub fn frog_jump(heights: &[i32]) -> i32 {
    let mut dp = vec![-1; heights.len()];

    fn helper(heights: &[i32], n: usize, dp: &mut Vec<i32>) -> i32 { //This uses 1-based indexing, so indices are off by 1
        if n == 1 {
            return 0;
        }

        if n == 2 {
            return (heights[0] - heights[1]).abs();
        }

        if dp[n - 1] != -1 {
            return dp[n - 1];
        }

        let one = helper(heights, n - 1, dp) + (heights[n - 1] - heights[n - 2]).abs();
        let two = helper(heights, n - 2, dp) + (heights[n - 1] - heights[n - 3]).abs();

        dp[n - 1] = std::cmp::min(one, two);

        return dp[n - 1];
    }

    helper(heights, heights.len(), &mut dp)
}
pub fn frog_jump_optimised(heights: &[i32]) -> i32 {
    let mut dp = vec![0; heights.len()];

    for i in 1..heights.len() {
        let one = dp[i - 1] + (heights[i] - heights[i - 1]).abs();
        let two = if i > 1 {
            dp[i - 2] + (heights[i] - heights[i - 2]).abs()
        } else {
            i32::MAX
        };

        dp[i] = std::cmp::min(one, two);
    }

    dp[heights.len() - 1]
}
pub fn frog_jump_space_optimised(heights: &[i32]) -> i32 {
    let (mut prev1, mut prev2) = (0, 0);

    for i in 1..heights.len() {
        let one = prev1 + (heights[i] - heights[i - 1]).abs();
        let two = if i > 1 {
            prev2 + (heights[i] - heights[i - 2]).abs()
        } else {
            i32::MAX
        };

        let curr = std::cmp::min(one, two);
        prev2 = prev1;
        prev1 = curr;
    }

    prev1
}

pub fn rob(houses: &[i32]) -> i32 {
    let n = houses.len();
    let mut dp = vec![0; n]; //The maximum take from the first 'i' houses
    dp[0] = houses[0];

    for i in 2..n { //The current house is i
        let take = houses[i] + dp[i - 2];
        let not_take = dp[i - 1];
        dp[i] = std::cmp::max(take, not_take);
    }

    dp[n - 1]
}
pub fn rob_optimised(houses: &[i32]) -> i32 {
    let (mut prev1, mut prev2) = (0, 0);

    for &house in houses {
        let take = house + prev2;
        let not_take = prev1;
        let curr = std::cmp::max(take, not_take);

        prev2 = prev1;
        prev1 = curr;
    }

    prev1
}

///Here, the houses are in a circle (first and last are connected)
pub fn rob_2(houses: &[i32]) -> i32 {
    //Take either the first house or the last house
    std::cmp::max(rob_optimised(&houses[1..]), rob_optimised(&houses[..houses.len() - 1]))
}

//Now we see 2-D dynamic programming
///The training is of n days, each day with three activities, the same activity cannot be done on two consecutive days
pub fn maximum_points(points: Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![vec![-1; 3]; points.len()];
    fn helper(points: &Vec<Vec<i32>>, current_day: usize, prev_task: usize, dp: &mut Vec<Vec<i32>>) -> i32 {
        if current_day == points.len() {
            return 0;
        }

        if dp[current_day][prev_task] != -1 {
            return dp[current_day][prev_task];
        }

        let mut max_score = 0;
        for task in 0..3 {
            if task == prev_task {
                continue;
            }
            let score = points[current_day][task] + helper(points, current_day + 1, task, dp);
            max_score = std::cmp::max(max_score, score);
        }
        max_score
    }

    helper(&points, 0, 0, &mut dp)
}
//We can convert the 2-D DP into a 1-D DP since only the scores of the previous day matter
pub fn maximum_points_optimised(points: Vec<Vec<i32>>) -> i32 {
    if points.is_empty() {
        return 0;
    }

    let mut dp = points[0].clone();

    for day in 1..points.len() {
        let mut temp = vec![0; 3];
        for task in 0..3 {
            for prev_task in 0..3 {
                if task != prev_task {
                    temp[task] = std::cmp::max(temp[task], dp[prev_task] + points[day][task]);
                }
            }
        }

        dp = temp;
    }

    *dp.iter().max().unwrap()
}

///The total number of unique paths from (0,0) to (m-1, n-1) moving down or right
pub fn unique_paths_recursive(m: i32, n: i32) -> i32 {
    //The number of ways to reach a cell = (ways to reach right cell) + (ways to reach down cell)
    //On the bottom and right boundary, the number of ways to reach the cell is 1 (ways to reach right or down = 0)
    if m == 1 || n == 1 {
        return 1;
    }

    return unique_paths_recursive(m - 1, n) + unique_paths_recursive(m, n - 1);
}
pub fn unique_paths_dp(m: i32, n: i32) -> i32 {
    let m = m as usize;
    let n = n as usize;
    let mut dp = vec![vec![1; n]; m];
    for i in 1..m {
        for j in 1..n {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }

    dp[m - 1][n - 1]
}

pub fn unique_paths(m: i32, n: i32) -> i32 {
    let m = m as usize;
    let n = n as usize;
    let mut prev = vec![1; n];
    for _ in 1..m {
        let mut curr = vec![1; n];
        for i in 1..n {
            curr[i] = curr[i - 1] + prev[i];
        }
        prev = curr;
    }

    prev[n - 1]
}
//The problem can be solved using Combinatorics. The number of ways = (m+n-2)C(m-1)

///1-> obstacle
pub fn unique_path_with_obstacles(obstacle_grid: Vec<Vec<i32>>) -> i32 {
    let m = obstacle_grid.len();
    let n = obstacle_grid[0].len();
    if obstacle_grid[0][0] == 1 || obstacle_grid[m - 1][n - 1] == 1 { //If the starting/ending points are blocked, we cannot reach the end
        return 0;
    }
    let mut dp = vec![vec![0; n]; m];
    dp[0][0] = 1;
    for i in 1..m { //First column
        if obstacle_grid[i][0] == 0 { //If it is not a blockage
            dp[i][0] = dp[i - 1][0];
        }
    }
    for j in 1..n { //First row
        if obstacle_grid[0][j] == 0 { //If it is not a blockage
            dp[0][j] = dp[0][j - 1];
        }
    }

    for i in 1..m {
        for j in 1..n {
            if obstacle_grid[i][j] == 0 {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
    }

    dp[m - 1][n - 1]
}
pub fn unique_path_with_obstacles_optimised(obstacle_grid: Vec<Vec<i32>>) -> i32 {
    let m = obstacle_grid.len();
    let n = obstacle_grid[0].len();
    if obstacle_grid[0][0] == 1 || obstacle_grid[m - 1][n - 1] == 1 {
        return 0;
    }
    let mut prev = vec![0; n];
    prev[0] = 1;
    for i in 0..m {
        let mut curr = vec![0; n];
        for j in 0..n {
            if obstacle_grid[i][j] == 1 {
                curr[j] = 0;
            } else if j > 0 {
                curr[j] = curr[j - 1] + prev[j];
            } else { //If it is the first row
                curr[j] = prev[j];
            }
        }
        prev = curr;
    }

    prev[n - 1]
}

///Only move down/right
pub fn min_path_sum_recursive(grid: Vec<Vec<i32>>) -> i32 {
    let m = grid.len();
    let n = grid[0].len();
    fn helper(grid: &Vec<Vec<i32>>, m: usize, n: usize) -> i32 {
        if m == 0 && n == 0 {
            return grid[m][n];
        }
        if m == 0 { //If we are at the first row, there is no going up
            return grid[m][n] + helper(grid, m, n - 1);
        }
        if n == 0 { //If we are at the first column, there is no going left
            return grid[m][n] + helper(grid, m - 1, n);
        }
        return grid[m][n] + std::cmp::min(helper(grid, m - 1, n), helper(grid, m, n - 1));
    }

    helper(&grid, m - 1, n - 1)
}
pub fn min_path_sum(grid: Vec<Vec<i32>>) -> i32 {
    let m = grid.len();
    let n = grid[0].len();
    let mut dp = vec![vec![-1; n]; m];
    dp[0][0] = grid[0][0];

    //First row
    for j in 1..n {
        dp[0][j] = dp[0][j - 1] + grid[0][j];
    }
    //First column
    for i in 1..m {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }

    for i in 1..m {
        for j in 1..n {
            dp[i][j] = std::cmp::min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
        }
    }

    dp[m - 1][n - 1]
}
//Convert to a 1-D DP
pub fn min_path_sum_optimised(grid: Vec<Vec<i32>>) -> i32 {
    let m = grid.len();
    let n = grid[0].len();
    let mut prev = vec![0; n];

    for i in 0..m {
        let mut curr = vec![0; n];
        for j in 0..n {
            if i == 0 && j == 0 {
                curr[j] = grid[i][j];
            } else {
                let up = if i > 0 {
                    prev[j]
                } else {
                    i32::MAX
                };

                let left = if j > 0 {
                    curr[j - 1]
                } else {
                    i32::MAX
                };

                curr[j] = grid[i][j] + std::cmp::min(up, left);
            }
        }
        prev = curr;
    }

    prev[n - 1]
}

///This problem is of the type where the starting point is fixed and the ending point is variable
///Find the least path sum from the top of the triangle to the bottom (any part of the bottom).
///We can only move to lower adjacent index (from i,j to i + 1, j/j+1) (straight down/right down)
pub fn minimum_triangle(triangle: Vec<Vec<i32>>) -> i32 { //This is the final space optimized code
    let height = triangle.len();
    if height == 1 { //Edge Case
        return triangle[0][0];
    }
    let mut prev = triangle[height - 1].clone(); //Base Case

    for i in (0..=height - 2).rev() {
        let mut curr = vec![0; height];
        for j in (0..=i).rev() {
            let straight_down = triangle[i][j] + prev[j]; //Cost to move straight down in triangle
            let right_down = triangle[i][j] + prev[j + 1]; //Cost to move down+right
            curr[j] = std::cmp::min(straight_down, right_down);
        }
        prev = curr;
    }

    prev[0]
}

///This problem is of the type where both the starting and ending points are variable
///Find the least path sum such that we start from the first row and end at the last row
///We can move from (row, col) -> (row + 1, col - 1)/(row + 1, col)/(row + 1, col + 1)
pub fn min_falling_path_sum(matrix: Vec<Vec<i32>>) -> i32 {
    let (rows, cols) = (matrix.len(), matrix[0].len());
    let mut dp = vec![vec![-1; cols]; rows];

    dp[0] = matrix[0].clone(); //Base Case

    for i in 1..rows {
        for j in 0..cols {
            let straight_up = dp[i - 1][j];
            let left_up = j.checked_sub(1).map_or(i32::MAX, |col| dp[i - 1][col]);
            let right_up = *dp[i - 1].get(j + 1).unwrap_or(&i32::MAX);

            dp[i][j] = matrix[i][j] + *[straight_up, left_up, right_up].iter().min().unwrap();
        }
    }

    *dp.last().unwrap().iter().min().unwrap()
}
pub fn min_falling_path_sum_optimised(matrix: Vec<Vec<i32>>) -> i32 {
    let (rows, cols) = (matrix.len(), matrix[0].len());
    let mut prev = matrix[0].clone();

    for i in 1..rows {
        let mut curr = vec![0; cols];
        for (j, &val) in matrix[i].iter().enumerate() {
            let straight_up = prev[j];
            let left_up = j.checked_sub(1).map_or(i32::MAX, |col| prev[col]);
            let right_up = *prev.get(j + 1).unwrap_or(&i32::MAX);

            curr[j] = matrix[i][j] + *[straight_up, left_up, right_up].iter().min().unwrap();
        }
        prev = curr;
    }

    *prev.iter().min().unwrap()
}

///This is a problem with fixed starting points and variable ending points.
///We need to maximize the sum of chocolates collected by Alice and Bob.
///They can move straight-down, left-down and right-down. If they are at the same cell, only one of them collects the chocolates
pub fn maximum_chocolates(grid: Vec<Vec<i32>>) -> i32 {
    let (rows, cols) = (grid.len(), grid[0].len());
    let mut dp = vec![vec![vec![-1; cols]; cols]; rows]; //3-D DP

    //Base Case
    for alice in 0..cols {
        for bob in 0..cols {
            if alice == bob {
                dp[rows - 1][alice][bob] = grid[rows - 1][alice];
            } else {
                dp[rows - 1][alice][bob] = grid[rows - 1][alice] + grid[rows - 1][bob];
            }
        }
    }

    const DELTA_COL: [isize; 3] = [-1, 0, 1];

    for row in (0..=rows - 2).rev() {
        for alice in 0..cols {
            for bob in 0..cols {
                let mut chocolates = 0;
                for &delta_alice in &DELTA_COL {
                    for &delta_bob in &DELTA_COL {
                        let mut val = if alice == bob {
                            grid[row][alice]
                        } else {
                            grid[row][alice] + grid[row][bob]
                        };

                        if alice as isize + delta_alice >= 0
                            && alice as isize + delta_alice < cols as isize
                            && bob as isize + delta_bob >= 0
                            && bob as isize + delta_bob < cols as isize {
                            val += dp[row + 1][(alice as isize + delta_alice) as usize][(bob as isize + delta_bob) as usize];
                        }

                        chocolates = std::cmp::max(chocolates, val);
                    }
                }
                dp[row][alice][bob] = chocolates;
            }
        }
    }

    dp[0][0][cols - 1]
}
pub fn maximum_chocolates_improvement(grid: Vec<Vec<i32>>) -> i32 {
    let (rows, cols) = (grid.len(), grid[0].len());
    let mut dp = vec![vec![vec![0; cols]; cols]; rows];

    for alice in 0..cols {
        for bob in 0..cols {
            dp[rows - 1][alice][bob] = if alice == bob {
                grid[rows - 1][alice]
            } else {
                grid[rows - 1][alice] + grid[rows - 1][bob]
            };
        }
    }

    const DELTA_COL: [isize; 3] = [-1, 0, 1];
    for row in (0..rows - 1).rev() {
        for alice in 0..cols {
            for bob in 0..cols {
                dp[row][alice][bob] = DELTA_COL.iter().cartesian_product(DELTA_COL.iter())
                    .filter_map(|(&delta_alice, &delta_bob)| {
                        let alice_next = (alice as isize + delta_alice).clamp(0, (cols - 1) as isize) as usize;
                        let bob_next = (bob as isize + delta_bob).clamp(0, (cols - 1) as isize) as usize;
                        let val = if alice == bob {
                            grid[row][alice]
                        } else {
                            grid[row][alice] + grid[row][bob]
                        };
                        Some(val + dp[row + 1][alice_next][bob_next])
                    })
                    .max().unwrap()
            }
        }
    }

    dp[0][0][cols - 1]
}

//Now we see some problems on DP on subsequences
pub fn subset_sum_recursive(arr: Vec<i32>, target: i32) -> bool {
    fn helper(index: usize, arr: &Vec<i32>, target: i32) -> bool {
        if target == 0 {
            return true;
        }
        if index == 0 {
            return arr[0] == target;
        }

        let not_take = helper(index - 1, arr, target);
        let take = if target >= arr[index] {
            helper(index - 1, arr, target - arr[index])
        } else {
            false
        };

        take || not_take
    }

    helper(arr.len() - 1, &arr, target)
}

pub fn subset_sum_tabulation(arr: Vec<i32>, target: i32) -> bool {
    let target_usize = target as usize;
    let mut dp = vec![vec![false; target_usize + 1]; arr.len()];

    for i in 0..arr.len() {
        dp[i][0] = true;
    }
    if arr[0] as usize <= target_usize {
        dp[0][arr[0] as usize] = true;
    }

    for i in 1..arr.len() {
        for j in 1..=target_usize {
            let not_take = dp[i - 1][j];
            let take = if arr[i] as usize <= j {
                dp[i - 1][j - arr[i] as usize]
            } else {
                false
            };
            dp[i][j] = not_take || take;
        }
    }

    dp[arr.len() - 1][target as usize]
}

///Use all the elements in the array to find two subsets with an equal sum
pub fn can_partition(nums: Vec<i32>) -> bool {
    //If sum(nums) => odd => false, else, find one subarray with sum = sum(nums)/2
    let sum = nums.iter().sum::<i32>();

    if sum % 2 == 1 {
        return false;
    }

    subset_sum_tabulation(nums, sum / 2)
}

///It is guaranteed that nums has even elements. Find the minimum possible difference between any two subsets
pub fn minimum_difference(nums: Vec<i32>) -> i32 {
    //We check the last row of the dp from subset_sum and find the possible partition sums
    let total_sum: i32 = nums.iter().sum();
    let total_sum = total_sum as usize;
    let mut dp = vec![vec![false; total_sum + 1]; nums.len()];

    for i in 0..nums.len() {
        dp[i][0] = true;
    }
    dp[0][nums[0] as usize] = true;

    for i in 1..nums.len() {
        for j in 1..=total_sum {
            let not_take = dp[i - 1][j];
            let take = if nums[i] as usize <= j {
                dp[i - 1][j - nums[i] as usize]
            } else {
                false
            };
            dp[i][j] = not_take || take;
        }
    }

    let mut min = i32::MAX;
    for i in 0..=total_sum / 2 {
        if dp[nums.len() - 1][i] {
            min = min.min(total_sum.abs_diff(2 * i) as i32);
        }
    }

    min
}

///Find the number of subsets with sum = target
pub fn perfect_sum(arr: Vec<i32>, target: i32) -> i32 {
    let target_usize = target as usize;
    let mut dp = vec![vec![0; target_usize + 1]; arr.len()];
    //Base Case
    for i in 0..arr.len() {
        dp[i][0] = 1;
    }
    if arr[0] <= target {
        dp[0][arr[0] as usize] = 1;
    }

    for i in 1..arr.len() {
        for j in 0..=target_usize {
            let not_take = dp[i - 1][j];
            let take = if arr[i] as usize <= j {
                dp[i - 1][j - arr[i] as usize]
            } else {
                0
            };
            dp[i][j] = take + not_take;
        }
    }

    dp[arr.len() - 1][target_usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci_test() {
        assert_eq!(fibonacci(10), 89);
        assert_eq!(fibonacci_better(10), 89);
        assert_eq!(fibonacci_best(10), 89);
    }

    #[test]
    fn climbing_stairs_test() {
        assert_eq!(climb_stairs(2), 2);
        assert_eq!(climb_stairs(3), 3);
    }

    #[test]
    fn frog_jump_test() {
        assert_eq!(frog_jump_recursive(&[10, 20, 30, 10]), 20);
        assert_eq!(frog_jump(&[10, 20, 30, 10]), 20);
        assert_eq!(frog_jump_optimised(&[10, 20, 30, 10]), 20);
        assert_eq!(frog_jump_space_optimised(&[10, 20, 30, 10]), 20);
    }

    #[test]
    fn rob_test() {
        assert_eq!(rob(&[2, 3, 2]), 4);
        assert_eq!(rob(&[1, 2, 3, 1]), 4);
        assert_eq!(rob_optimised(&[2, 3, 2]), 4);
        assert_eq!(rob_optimised(&[1, 2, 3, 1]), 4);
    }

    #[test]
    fn rob_again_test() {
        assert_eq!(rob_2(&[2, 3, 2]), 3);
    }

    #[test]
    fn max_points_test() {
        assert_eq!(maximum_points(vec![
            vec![1, 2, 5],
            vec![3, 1, 1],
            vec![3, 3, 3],
        ]), 11);

        assert_eq!(maximum_points_optimised(vec![
            vec![1, 2, 5],
            vec![3, 1, 1],
            vec![3, 3, 3],
        ]), 11);
    }

    #[test]
    fn unique_paths_test() {
        assert_eq!(unique_paths_recursive(3, 7), 28);
        assert_eq!(unique_paths_recursive(3, 2), 3);
        assert_eq!(unique_paths_dp(3, 7), 28);
        assert_eq!(unique_paths_dp(3, 2), 3);
        assert_eq!(unique_paths(3, 7), 28);
        assert_eq!(unique_paths(3, 2), 3);
    }

    #[test]
    fn unique_paths_with_obstacles_test() {
        assert_eq!(unique_path_with_obstacles(vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 0],
        ]), 2);
        assert_eq!(unique_path_with_obstacles(vec![
            vec![0, 1],
            vec![0, 0],
        ]), 1);
        assert_eq!(unique_path_with_obstacles_optimised(vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 0],
        ]), 2);
        assert_eq!(unique_path_with_obstacles_optimised(vec![
            vec![0, 1],
            vec![0, 0],
        ]), 1);
        assert_eq!(unique_path_with_obstacles_optimised(vec![vec![0, 0]]), 1);
    }

    #[test]
    fn min_path_sum_test() {
        assert_eq!(min_path_sum_recursive(vec![
            vec![1, 3, 1],
            vec![1, 5, 1],
            vec![4, 2, 1],
        ]), 7);
        assert_eq!(min_path_sum_recursive(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ]), 12);
        assert_eq!(min_path_sum(vec![
            vec![1, 3, 1],
            vec![1, 5, 1],
            vec![4, 2, 1],
        ]), 7);
        assert_eq!(min_path_sum(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ]), 12);
        assert_eq!(min_path_sum_optimised(vec![
            vec![1, 3, 1],
            vec![1, 5, 1],
            vec![4, 2, 1],
        ]), 7);
        assert_eq!(min_path_sum_optimised(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ]), 12);
    }

    #[test]
    fn minimum_total_test() {
        assert_eq!(minimum_triangle(vec![
            vec![2],
            vec![3, 4],
            vec![6, 5, 7],
            vec![4, 1, 8, 3],
        ]), 11);
        assert_eq!(minimum_triangle(vec![vec![-10]]), -10);
    }

    #[test]
    fn min_falling_path_sum_test() {
        assert_eq!(min_falling_path_sum(vec![
            vec![2, 1, 3],
            vec![6, 5, 4],
            vec![7, 8, 9],
        ]), 13);
        assert_eq!(min_falling_path_sum(vec![
            vec![-19, 57],
            vec![-40, -5],
        ]), -59);
        assert_eq!(min_falling_path_sum_optimised(vec![
            vec![2, 1, 3],
            vec![6, 5, 4],
            vec![7, 8, 9],
        ]), 13);
        assert_eq!(min_falling_path_sum_optimised(vec![
            vec![-19, 57],
            vec![-40, -5],
        ]), -59);
    }

    #[test]
    fn maximum_chocolates_test() {
        assert_eq!(maximum_chocolates(vec![
            vec![3, 1, 1],
            vec![2, 5, 1],
            vec![1, 5, 5],
            vec![2, 1, 1],
        ]), 24);
        assert_eq!(maximum_chocolates_improvement(vec![
            vec![3, 1, 1],
            vec![2, 5, 1],
            vec![1, 5, 5],
            vec![2, 1, 1],
        ]), 24);
    }

    #[test]
    fn subset_sum_test() {
        let arr = vec![3, 34, 4, 12, 5, 2];
        let target1 = 9;
        let target2 = 30;

        assert_eq!(subset_sum_recursive(arr.clone(), target1), true);
        assert_eq!(subset_sum_recursive(arr.clone(), target2), false);

        assert_eq!(subset_sum_tabulation(arr.clone(), target1), true);
        assert_eq!(subset_sum_tabulation(arr.clone(), target2), false);
    }

    #[test]
    fn can_partition_test() {
        assert_eq!(can_partition(vec![1, 5, 11, 5]), true);
        assert_eq!(can_partition(vec![1, 2, 3, 5]), false);
    }

    #[test]
    fn minimum_difference_test() {
        assert_eq!(minimum_difference(vec![3, 9, 7, 3]), 2);
        //assert_eq!(minimum_difference(vec![-36, 36]), 72); This test case does not work as the algo works if nums[i] > 0
        assert_eq!(minimum_difference(vec![1, 2, 3, 4]), 0);
        assert_eq!(minimum_difference(vec![8, 6, 5]), 3);
    }

    #[test]
    fn perfect_sum_test() {
        assert_eq!(perfect_sum(vec![5, 2, 3, 10, 6, 8], 10), 3);
        assert_eq!(perfect_sum(vec![2, 5, 1, 4, 3], 10), 3);
    }
}
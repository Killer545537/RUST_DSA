/* Dynamic Programming (DP) is a method to used to solve problems by dividing them into smaller, easier sub-problems.
The two types of approaches to DP, Top-Down(Memoization) and Bottom-Up(Tabulation)
In memoization, we start from the top and recurse down while storing the results to the sub-problems
In tabulation, we start from the bottom and recurse up using the results of the sub-problems  */

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
//Even this can be optimised, since only the last two values are being used, only use them

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
    let mut dp = vec![0; n]; //The maximum take from the first i houses
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

///Here, the houses are in a circle (1st and last are connected)
pub fn rob_2(houses: &[i32]) -> i32 {
    //Take either the first house or the last house
    std::cmp::max(rob_optimised(&houses[1..]), rob_optimised(&houses[..houses.len() - 1]))
}

//Now we see 2-D dynamic programming
///The training is of n days, each day with 3 activities, same activity cannot be done on two consecutive days
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
//TODO- Convert into 1-D

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
    }
}
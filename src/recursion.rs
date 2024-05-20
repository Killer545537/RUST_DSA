use std::collections::HashSet;
use std::os::unix::raw::mode_t;
use crate::is_palindrome;

pub fn my_atoi(s: String) -> i32 {
    let s = s.trim(); //Remove all leading and trailing spaces
    let (s, sign) = match s.strip_prefix('-') { //Get the sign of the integer
        Some(s) => (s, -1),
        None => (s.strip_prefix('+').unwrap_or(s), 1)
    };

    s.chars().map(|c| c.to_digit(10))
        .take_while(|d| d.is_some())
        .flatten()
        .fold(0, |acc, digit| {
            acc.saturating_mul(10).saturating_add(sign * digit as i32)
        })
}

pub fn my_atoi_rec(s: String) -> i32 {
    let s = s.trim();
    let (s, sign) = match s.strip_prefix('-') { //Get the sign of the integer
        Some(s) => (s, -1),
        None => (s.strip_prefix('+').unwrap_or(s), 1)
    };

    fn my_atoi_rec_helper(s: &str, index: usize, sign: i32, acc: i32) -> i32 {
        if index >= s.len() {
            return acc;
        }

        match s.chars().nth(index).unwrap().to_digit(10) {
            Some(digit) => {
                let acc = acc.saturating_mul(10).saturating_add(sign * digit as i32);
                my_atoi_rec_helper(s, index + 1, sign, acc)
            }
            None => acc
        }
    }

    my_atoi_rec_helper(s, 0, sign, 0)
}

pub fn my_pow(x: f64, n: i32) -> f64 {
    let mut ans = 1.0;
    let mut nn = n as i64; //Since n go out of bounds of i32
    let mut x = x;
    if nn < 0 {
        nn *= -1;
    }

    while nn > 0 {
        if nn % 2 == 1 {
            ans *= x;
            nn -= 1;
        } else { //x^n = (x^2)^(n/2)
            x *= x;
            nn /= 2;
        }
    }

    if n < 0 {
        ans = 1.0 / ans; //Take the reciprocal
    }

    ans
}

pub fn generate_parenthesis(n: i32) -> Vec<String> {
    let mut ans = Vec::new();
    let mut s = String::new();
    fn generate(ans: &mut Vec<String>, s: &mut String, open: i32, close: i32) { //open and close are the number of remaining brackets left
        if open == 0 && close == 0 { //If all are exhausted then it is a valid string
            ans.push(s.clone());
        }

        if open > 0 { //If any open left
            s.push('(');
            generate(ans, s, open - 1, close);
            s.pop(); //Backtracking step
        }

        if close > 0 {
            if open < close { //If the number of open in the string are more, we must use a closing bracket
                s.push(')');
                generate(ans, s, open, close - 1);
                s.pop();
            }
        }
    }

    generate(&mut ans, &mut s, n, n);
    ans
}

pub fn combination_sum(candidates: &[i32], target: i32) -> Vec<Vec<i32>> { //Here, elements can be taken multiple times
    let mut ans = Vec::new();
    let mut possible = Vec::new();
    fn helper(index: usize, target: i32, candidates: &[i32], possible: &mut Vec<i32>, ans: &mut Vec<Vec<i32>>) {
        if index == candidates.len() {
            if target == 0 {
                ans.push(possible.clone());
            }
            return;
        }

        if candidates[index] <= target {
            possible.push(candidates[index]);
            helper(index, target - candidates[index], candidates, possible, ans);
            possible.pop();
        }

        helper(index + 1, target, candidates, possible, ans);
    }

    helper(0, target, candidates, &mut possible, &mut ans);

    ans
}

pub fn combination_sum_again(candidates: &Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    let mut candidates = candidates.clone();
    candidates.sort();
    let mut ans: Vec<Vec<i32>> = Vec::new();
    let mut possible: Vec<i32> = Vec::new();
    fn helper(index: usize, target: i32, candidates: &Vec<i32>, possible: &mut Vec<i32>, ans: &mut Vec<Vec<i32>>) {
        if target == 0 {
            ans.push(possible.clone());
            return;
        }

        for i in index..candidates.len() {
            if i > index && candidates[i] == candidates[i - 1] {
                continue;
            }

            if candidates[i] > target {
                break;
            }

            possible.push(candidates[i]);
            helper(i + 1, target - candidates[i], candidates, possible, ans);
            possible.pop();
        }
    }

    helper(0, target, &candidates, &mut possible, &mut ans);

    ans
}

pub fn subset_sum(arr: &[i32]) -> Vec<i32> {
    let mut ans: Vec<i32> = Vec::with_capacity(2usize.pow(arr.len() as u32));
    fn helper(index: usize, sum: i32, arr: &[i32], ans: &mut Vec<i32>) {
        if index == arr.len() {
            ans.push(sum);
            return;
        }

        helper(index + 1, sum + arr[index], arr, ans);
        helper(index + 1, sum, arr, ans);
    }

    helper(0, 0, arr, &mut ans);
    ans.sort();
    ans
}

pub fn subsets_with_dup(arr: &[i32]) -> Vec<Vec<i32>> {
    let mut arr = arr.clone().to_vec();
    arr.sort();
    let mut ans: Vec<Vec<i32>> = Vec::with_capacity(2usize.pow(arr.len() as u32));
    let mut possible: Vec<i32> = Vec::with_capacity(arr.len());
    fn helper(index: usize, arr: &Vec<i32>, possible: &mut Vec<i32>, ans: &mut Vec<Vec<i32>>) {
        ans.push(possible.clone());

        for i in index..arr.len() {
            if i > index && arr[i] == arr[i - 1] {
                continue;
            }

            possible.push(arr[i]);
            helper(i + 1, arr, possible, ans);
            possible.pop();
        }
    }

    helper(0, &arr, &mut possible, &mut ans);

    ans
}

pub fn combination_sum_returns(k: i32, n: i32) -> Vec<Vec<i32>> {
    let arr: Vec<i32> = (1..10).collect();
    let mut ans: Vec<Vec<i32>> = Vec::new();
    let mut possible: Vec<i32> = Vec::new();
    fn helper(index: usize, arr: &Vec<i32>, k: usize, n: i32, sum: i32, possible: &mut Vec<i32>, ans: &mut Vec<Vec<i32>>) {
        if possible.len() == k {
            if sum == n {
                ans.push(possible.clone());
            }
            return;
        }

        if index >= arr.len() {
            return;
        }

        if arr[index] <= n - sum {
            possible.push(arr[index]);
            helper(index + 1, arr, k, n, sum + arr[index], possible, ans);
            possible.pop();
        }
        helper(index + 1, arr, k, n, sum, possible, ans);
    }

    helper(0, &arr, k as usize, n, 0, &mut possible, &mut ans);

    ans
}

pub fn permute(arr: &[i32]) -> Vec<Vec<i32>> {
    //This can also be done using a map storing if the element has occurred in the permutation
    let mut ans: Vec<Vec<i32>> = Vec::with_capacity((1..=arr.len()).product::<usize>());
    let mut perm = arr.clone().to_vec();
    fn helper(index: usize, perm: &mut Vec<i32>, ans: &mut Vec<Vec<i32>>) {
        if index == perm.len() {
            ans.push(perm.clone());
            return;
        }

        for i in index..perm.len() {
            perm.swap(index, i);
            helper(index + 1, perm, ans);
            perm.swap(index, i);
        }
    }

    helper(0, &mut perm, &mut ans);

    ans
}


//Find all partitions which result in the substrings being palindromes
pub fn partition(s: String) -> Vec<Vec<String>> {
    let mut ans: Vec<Vec<String>> = Vec::new();
    let mut path: Vec<String> = Vec::new();
    fn is_palindrome(s: &str) -> bool {
        s.chars().eq(s.chars().rev())
    }
    fn helper(index: usize, s: &String, path: &mut Vec<String>, ans: &mut Vec<Vec<String>>) {
        if index == s.len() {
            ans.push(path.clone());
            return;
        }

        for i in index..s.len() {
            if is_palindrome(&s[index..=i]) {
                path.push(s[index..=i].to_string());
                helper(i + 1, s, path, ans);
                path.pop();
            }
        }
    }

    helper(0, &s, &mut path, &mut ans);

    ans
}

pub fn solve_n_queens(n: usize) -> Vec<Vec<Vec<bool>>> {
    let mut ans: Vec<Vec<Vec<bool>>> = Vec::new();
    let mut board = vec![vec![false; n]];
    fn solve(n: usize, col: usize, board: &mut Vec<Vec<bool>>, ans: &mut Vec<Vec<Vec<bool>>>) {
        fn is_safe(n: usize, row: usize, col: usize, board: &Vec<Vec<bool>>) -> bool {
            //Check the left side of the current row
            for i in 0..col {
                if board[row][i] {
                    return false;
                }
            }
            //Check the upper left diagonal
            let (mut i, mut j) = (row, col);
            while i > 0 && j > 0 {
                if board[i - 1][j - 1] {
                    return false;
                }
                i -= 1;
                j -= 1;
            }
            //Check the lower left diagonal
            i = row;
            j = col;
            while j > 0 && i < n - 1 {
                if board[i + 1][j - 1] {
                    return false;
                }
                i += 1;
                j -= 1;
            }

            true
        }

        if col == n {
            ans.push(board.clone());
            return;
        }

        for row in 0..n {
            if is_safe(n, row, col, board) {
                board[row][col] = true;
                solve(n, col + 1, board, ans);
                board[row][col] = false;
            }
        }
    }

    solve(n, 0, &mut board, &mut ans);

    ans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atoi_rec_test() {
        assert_eq!(my_atoi_rec("42".to_string()), 42);
        assert_eq!(my_atoi_rec(" -042".to_string()), -42);
    }

    #[test]
    fn pow_pow() {
        assert_eq!(my_pow(2.0, 10), 1024.0);
        assert_eq!(my_pow(2.0, -2), 0.25);
    }

    #[test]
    fn parenthesis_test() {
        assert_eq!(generate_parenthesis(1), vec!["()"]);
        assert_eq!(generate_parenthesis(3), vec!["((()))", "(()())", "(())()", "()(())", "()()()"]);
    }

    #[test]
    fn combination_sum_test() {
        assert_eq!(combination_sum(&[2, 3, 6, 7], 7), vec![vec![2, 2, 3], vec![7]]);
        assert_eq!(combination_sum(&[2, 3, 5], 8), vec![
            vec![2, 2, 2, 2],
            vec![2, 3, 3],
            vec![3, 5],
        ]);
    }

    #[test]
    fn combination_sum_again_test() {
        assert_eq!(combination_sum_again(&[10, 1, 2, 7, 6, 1, 5].to_vec(), 8), vec![
            vec![1, 1, 6],
            vec![1, 2, 5],
            vec![1, 7],
            vec![2, 6],
        ]);
    }

    #[test]
    fn subset_sum_test() {
        assert_eq!(subset_sum(&[2, 3]), vec![0, 2, 3, 5]);
        assert_eq!(subset_sum(&[5, 2, 1]), vec![0, 1, 2, 3, 5, 6, 7, 8]);
    }

    #[test]
    fn subset_dup_test() {
        assert_eq!(subsets_with_dup(&[1, 2, 2]), vec![vec![], vec![1], vec![1, 2], vec![1, 2, 2], vec![2], vec![2, 2]]);
    }

    #[test]
    fn combination_sum_returns_test() {
        assert_eq!(combination_sum_returns(3, 7), vec![vec![1, 2, 4]]);
    }

    #[test]
    fn partition_test() {
        assert_eq!(partition("aab".to_string()), vec![
            vec!["a".to_string(), "a".to_string(), "b".to_string()],
            vec!["aa".to_string(), "b".to_string()],
        ]);
    }
}
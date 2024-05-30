//A greedy approach is one which makes the best choice available at the current moment (no consideration of future events)

//Here, greed is the greed factor of each child and cookies is the greed factor that it can satisfy
//For a child to be content cookie >= greed
pub fn find_content_children(greed: &[i32], cookies: &[i32]) -> i32 {
    let mut greed = greed.to_vec();
    greed.sort();
    let mut cookies = cookies.to_vec();
    cookies.sort();

    let (mut l, mut r) = (0, 0); //l points to greed and r to cookies
    while l < greed.len() && r < cookies.len() {
        if cookies[r] >= greed[l] {
            l += 1; //If the greed is satisfied move to the next child and cookie
        }
        r += 1; //If it is not satisfied, move to the next cookie
    }

    l as i32
}

pub fn lemonade_change(bills: Vec<i32>) -> bool {
    let (mut five, mut ten) = (0, 0); //There is no need to count 20 bills

    for bill in bills {
        match bill {
            5 => five += 1,
            10 => {
                if five > 0 {
                    five -= 1;
                    ten += 1;
                } else {
                    return false;
                }
            }
            20 => {
                if ten > 0 && five > 0 {
                    ten -= 1;
                    five -= 1;
                } else if five >= 3 {
                    five -= 3;
                } else {
                    return false;
                }
            }
            _ => return false,
        }
    }

    true
}

//arr[i] is the number of places we can jump, i.e. from i to i + arr[i]
pub fn can_jump(arr: Vec<i32>) -> bool {
    //The only obstacle is 0, so at every index we track the maximum element we can reach
    let mut max_index = 0;

    for (ind, &jump) in arr.iter().enumerate() {
        match ind > max_index {
            true => return false,
            false => max_index = std::cmp::max(max_index, ind + jump as usize)
        }

        if max_index >= arr.len() - 1 { //If the array can be completed
            return true;
        }
    }

    true
}

pub fn jump(arr: Vec<i32>) -> i32 {
    let (mut jumps, mut l, mut r) = (0, 0, 0); //l -> r forms a range which can be reached in a jump

    while r < arr.len() - 1 {
        let farthest = (l..=r).map(|i| i + arr[i] as usize).max().unwrap();

        l = r + 1;
        r = farthest;
        jumps += 1;
    }

    jumps
}

#[derive(Debug)]
struct Job {
    id: usize,
    dead: usize,
    profit: i8,
}

///We need to find the number of jobs done and the maximum profit. Only 1 job can be done each day
pub fn job_scheduling(mut jobs: Vec<Job>) -> (usize, i8) {
    //We need to delay the job as much as profit, and complete the maximum profit job with the first deadline
    jobs.sort_by(|a, b| b.profit.cmp(&a.profit));
    let (mut total_profit, mut count_jobs) = (0, 0);
    let max_deadline = jobs.iter().max_by_key(|x| x.dead).unwrap().dead;
    let mut hash = vec![None; max_deadline + 1];

    for job in jobs {
        for day in (1..=job.dead).rev() {
            if hash[day].is_none() {
                count_jobs += 1;
                total_profit += job.profit;
                hash[day] = Some(job.id);
                break;
            }
        }
    }

    (count_jobs, total_profit)
}

pub fn max_meetings(start: Vec<i32>, end: Vec<i32>) -> (Vec<usize>, i32) {
    let mut timings: Vec<(usize, (i32, i32))> = start.into_iter().zip(end).enumerate().collect(); //position. start_time, end_time
    //Sort by end timings and then by start timings
    timings.sort_by(|&a, &b| a.1.1.cmp(&b.1.1).then_with(|| a.1.0.cmp(&b.1.0)));

    let (mut count, mut free_time) = (1, timings[0].1.1); //Start with the first meeting
    let mut order = vec![0];

    for &meet in timings.iter().skip(1) {
        if meet.1.0 > free_time { //If the meet starts after the free_time
            count += 1;
            free_time = meet.1.1;
            order.push(meet.0);
        }
    }

    (order, count)
}

///Return the minimum number of intervals to be removed
pub fn erase_overlap_intervals(mut intervals: Vec<(i32, i32)>) -> i32 {
    intervals.sort_by(|&a, &b| a.1.cmp(&b.1));

    let (mut count, mut free_time) = (0, intervals[0].1);
    for interval in intervals.into_iter().skip(1) {
        if interval.0 < free_time {
            count += 1;
        } else {
            free_time = interval.1;
        }
    }

    count
}

///Insert new_interval in intervals(non-overlapping) such that the resultant array is also non-overlapping
pub fn insert_intervals(intervals: Vec<(i32, i32)>, mut new_interval: (i32, i32)) -> Vec<(i32, i32)> {
    let mut ans: Vec<(i32, i32)> = Vec::new();
    let mut i = 0;

    //Part to the left of the new_interval
    while i < intervals.len() && intervals[i].1 < new_interval.0 {
        ans.push(intervals[i]);
        i += 1;
    }

    //Final merged interval (if needed)
    while i < intervals.len() && intervals[i].0 <= new_interval.1 {
        //Mutate the interval itself to the new merged interval
        new_interval.0 = std::cmp::min(new_interval.0, intervals[i].0);
        new_interval.1 = std::cmp::max(new_interval.1, intervals[i].1);
        i += 1;
    }
    ans.push(new_interval);

    //Part to the right of the new_interval
    while i < intervals.len() {
        ans.push(intervals[i]);
        i += 1;
    }

    ans
}

pub fn platforms(mut arrival: Vec<i32>, mut departure: Vec<i32>) -> i32 {
    arrival.sort();
    departure.sort();
    let (mut i, mut j) = (0, 0);
    let (mut platforms_needed, mut max_platforms) = (0, 0);

    while i < arrival.len() && j < departure.len() {
        if arrival[i] <= departure[j] {
            platforms_needed += 1;
            i += 1;
        } else {
            platforms_needed -= 1;
            j += 1;
        }
        max_platforms = std::cmp::max(max_platforms, platforms_needed);
    }

    max_platforms
}

///Each child must get more than 1 candy and the child with higher rating gets more
pub fn candy(ratings: Vec<i32>) -> i32 {
    let mut left_neighbour = vec![0; ratings.len()];
    left_neighbour[0] = 1;
    let mut right_neighbour = vec![0; ratings.len()];
    right_neighbour[ratings.len() - 1] = 1;
    for i in 1..ratings.len() {
        if ratings[i] > ratings[i - 1] {
            left_neighbour[i] = left_neighbour[i - 1] + 1;
        } else {
            left_neighbour[i] = 1;
        }
    }

    for i in (0..ratings.len() - 1).rev() {
        if ratings[i] > ratings[i + 1] {
            right_neighbour[i] = right_neighbour[i + 1] + 1;
        } else {
            right_neighbour[i] = 1;
        }
    }

    left_neighbour.into_iter().zip(right_neighbour).map(|(a, b)| std::cmp::max(a, b)).sum()
}

//This approach uses O(2N) extra space
pub fn candy_best(ratings: Vec<i32>) -> i32 {
    let mut sum = 1;
    let mut i = 1;

    while i < ratings.len() {
        //Constant
        if ratings[i] == ratings[i - 1] {
            sum = sum + 1;
            i += 1;
            continue;
        }
        let mut peak = 1;
        //Increasing
        while i < ratings.len() && ratings[i] > ratings[i - 1] {
            peak += 1;
            sum += peak;
            i += 1;
        }
        let mut down = 1;
        //Decreasing
        while i < ratings.len() && ratings[i] < ratings[i - 1] {
            sum += down;
            down += 1;
            i += 1;
        }

        if down > peak {
            sum += down - peak;
        }
    }

    sum
}

///Store the maximum possible profit (items = (profit, weight)) where each item can be broken down
pub fn fractional_knapsack(mut weight: f64, mut items: Vec<(f64, f64)>) -> f64 {
    items.sort_by(|&a, &b| (b.0 / b.1).partial_cmp(&(a.0 / a.1)).unwrap());
    let mut profit = 0.0;
    for i in items {
        if i.1 <= weight {
            profit += i.0;
            weight -= i.1;
        } else {
            profit += (i.0 / i.1) * weight;
            break;
        }
    }

    profit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_children_test() {
        assert_eq!(find_content_children(&[1, 2, 3], &[1, 1]), 1);
        assert_eq!(find_content_children(&[1, 2], &[1, 2]), 2);
    }

    #[test]
    fn change_test() {
        assert_eq!(lemonade_change(vec![5, 5, 5, 10, 20]), true);
        assert_eq!(lemonade_change(vec![5, 5, 10, 10, 20]), false);
    }

    #[test]
    fn jumper_test() {
        assert_eq!(can_jump(vec![2, 3, 1, 1, 4]), true);
        assert_eq!(can_jump(vec![3, 2, 1, 0, 4]), false);
    }

    #[test]
    fn another_jumper_test() {
        assert_eq!(jump(vec![2, 3, 1, 1, 4]), 2);
        assert_eq!(jump(vec![2, 3, 0, 1, 4]), 2);
    }

    #[test]
    fn job_scheduling_test() {
        let jobs = vec![
            Job { id: 1, dead: 4, profit: 20 },
            Job { id: 2, dead: 1, profit: 10 },
            Job { id: 3, dead: 1, profit: 40 },
            Job { id: 4, dead: 1, profit: 30 },
        ];
        assert_eq!(job_scheduling(jobs), (2, 60));
        let jobs = vec![
            Job { id: 1, dead: 2, profit: 100 },
            Job { id: 2, dead: 1, profit: 19 },
            Job { id: 3, dead: 2, profit: 27 },
            Job { id: 4, dead: 1, profit: 25 },
            Job { id: 5, dead: 1, profit: 15 },
        ];
        assert_eq!(job_scheduling(jobs), (2, 127))
    }

    #[test]
    fn max_meetings_test() {
        assert_eq!(max_meetings(vec![1, 3, 0, 5, 8, 5], vec![2, 4, 6, 7, 9, 9]), (vec![0, 1, 3, 4], 4));
        assert_eq!(max_meetings(vec![10, 12, 20], vec![20, 25, 30]), (vec![0], 1));
    }

    #[test]
    fn erase_overlap_intervals_test() {
        assert_eq!(erase_overlap_intervals(vec![(1, 2), (2, 3), (3, 4), (1, 3)]), 1);
        assert_eq!(erase_overlap_intervals(vec![(1, 2), (1, 2), (1, 2)]), 2);
        assert_eq!(erase_overlap_intervals(vec![(1, 2), (2, 3)]), 0)
    }

    #[test]
    fn insert_intervals_test() {
        assert_eq!(insert_intervals(vec![(1, 3), (6, 9)], (2, 5)), vec![(1, 5), (6, 9)]);
        assert_eq!(insert_intervals(vec![(1, 2), (3, 5), (6, 7), (8, 10), (12, 16)], (4, 8)), vec![(1, 2), (3, 10), (12, 16)]);
    }

    #[test]
    fn platforms_test() {
        assert_eq!(platforms(vec![0900, 0940, 0950, 1100, 1500, 1800], vec![0910, 1200, 1120, 1130, 1900, 2000]), 3);
        assert_eq!(platforms(vec![0900, 1100, 1235], vec![1000, 1200, 1240]), 1);
    }

    #[test]
    fn candy_test() {
        assert_eq!(candy(vec![1, 0, 2]), 5);
        assert_eq!(candy_best(vec![1, 0, 2]), 5);
    }

    #[test]
    fn fractional_knapsack_test() {
        assert_eq!(fractional_knapsack(50.0, vec![(60.0,10.0), (100.0,20.0), (120.0,30.0)]), 240.0);
    }
}
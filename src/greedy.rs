//A greedy approach is one which makes the best choice available at the current moment (no consideration of future events)

use itertools::rev;

//Here, greed is the greed factor of each child and cookies is the greed factor that it can satisfy
//For a child to be content cookie >= greed
pub fn find_content_children(greed: &[i32], cookies: &[i32]) -> i32 {
    let mut greed = greed.to_vec();
    greed.sort();
    let mut cookies = cookies.to_vec();
    cookies.sort();

    let (mut l,  mut r) = (0, 0); //l points to greed and r to cookies
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
        let mut farthest = (l..=r).map(|i| i + arr[i] as usize).max().unwrap();

        l = r + 1;
        r = farthest;
        jumps += 1;
    }

    jumps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_children_test() {
        assert_eq!(find_content_children(&[1,2,3], &[1,1]), 1);
        assert_eq!(find_content_children(&[1,2], &[1,2]), 2);
    }

    #[test]
    fn change_test() {
        assert_eq!(lemonade_change(vec![5,5,5,10,20]), true);
        assert_eq!(lemonade_change(vec![5,5,10,10,20]), false);
    }

    #[test]
    fn jumper_test() {
        assert_eq!(can_jump(vec![2,3,1,1,4]), true);
        assert_eq!(can_jump(vec![3,2,1,0,4]), false);
    }

    #[test]
    fn another_jumper_test() {
        assert_eq!(jump(vec![2,3,1,1,4]), 2);
        assert_eq!(jump(vec![2,3,0,1,4]), 2);
    }
}
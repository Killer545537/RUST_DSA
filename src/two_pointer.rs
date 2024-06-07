///k steps. Cards can be picked from either the end or beginning
pub fn max_score(card_points: Vec<i32>, k: usize) -> i32 {
    if card_points.is_empty() || k > card_points.len() {
        return 0;
    }

    let (mut lsum, mut rsum) = (card_points[..k].iter().sum::<i32>(), 0);
    let mut max_sum = lsum;
    let mut r_ptr = card_points.len() - 1;
    for i in (0..=k - 1).rev() {
        lsum -= card_points[i];
        if r_ptr > 0 {
            rsum += card_points[r_ptr];
            r_ptr -= 1;
        } else {
            break;
        }


        max_sum = std::cmp::max(lsum + rsum, max_sum);
    }

    max_sum
}

pub fn max_score_idk(card_points: Vec<i32>, k: usize) -> i32 {
    let n = card_points.len();
    let k = k.min(n);

    let mut lsum: i32 = card_points[..k].iter().sum();
    let mut rsum = 0;
    let mut max_sum = lsum;

    for i in (0..k).rev() {
        lsum = lsum.saturating_sub(card_points[i]);
        rsum += card_points.get(card_points.len() - k + i).unwrap_or(&0);
        max_sum = max_sum.max(lsum + rsum);
    }

    max_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_score_test() {
        assert_eq!(max_score_idk(vec![1, 2, 3, 4, 5, 6, 1], 3), 12);
        assert_eq!(max_score_idk(vec![2, 2, 2], 2), 4);
        assert_eq!(max_score_idk(vec![9, 7, 7, 9, 7, 7, 9], 7), 55);
    }
}
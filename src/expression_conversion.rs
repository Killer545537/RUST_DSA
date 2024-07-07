//Expression Conversion

//Infix expressions are those where the binary operator is between the operands. E.g., 1 + 2.
//Postfix expressions are those where the operator is after the operands. E.g., 1 2 +.
//Prefix expressions are those where the operator is before the operands. E.g., + 1 2.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExpressionError {
    #[error("Invalid Operator")]
    InvalidOperator,

    #[error("Invalid Expression String")]
    InvalidExpressionString,

    #[error("Invalid Expression Conversion")]
    InvalidExpressionConversion,
}

#[derive(PartialEq, Eq, Clone)]
enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl Operator {
    fn precedence(&self) -> i32 {
        match self {
            Operator::Add | Operator::Subtract => 1,
            Operator::Multiply | Operator::Divide => 2
        }
    }
}

impl PartialOrd for Operator {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.precedence().partial_cmp(&other.precedence())
    }
}

impl Ord for Operator {
    fn cmp(&self, other: &Self) -> Ordering {
        self.precedence().cmp(&other.precedence())
    }
}

impl TryFrom<char> for Operator {
    type Error = ExpressionError;

    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            '+' => Ok(Operator::Add),
            '-' => Ok(Operator::Subtract),
            '*' => Ok(Operator::Multiply),
            '/' => Ok(Operator::Divide),
            _ => Err(ExpressionError::InvalidOperator)
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self {
            Operator::Add => write!(f, "+"),
            Operator::Subtract => write!(f, "-"),
            Operator::Multiply => write!(f, "*"),
            Operator::Divide => write!(f, "/")
        }
    }
}

pub struct Infix;
pub struct Prefix;
pub struct Postfix;

#[derive(Clone)]
enum ExpressionElement {
    Operator(Operator),
    Operand(char),
    Parenthesis(char),
}

pub enum ExpressionType {
    Infix,
    Postfix,
    Prefix,
}

pub struct Expression<T> {
    expression: Vec<ExpressionElement>,
    expression_type: T,
}

//NOTE- An expression may evaluate to multiple equivalent expressions
impl Expression<Infix> {
    pub fn infix_to_postfix(&self) -> Result<Expression<Postfix>, ExpressionError> {
        let mut stack = VecDeque::new();
        let mut postfix: Vec<ExpressionElement> = Vec::new();

        for element in &self.expression {
            match element {
                ExpressionElement::Operator(op) => {
                    while stack.back().map_or(false, |top| {
                        if let ExpressionElement::Operator(top_op) = top {
                            top_op >= op
                        } else {
                            false
                        }
                    }) {
                        postfix.push(stack.pop_back().unwrap());
                    }
                    stack.push_back((*element).clone());
                }
                ExpressionElement::Operand(_) => postfix.push(element.clone()),
                ExpressionElement::Parenthesis(c) => {
                    if *c == '(' {
                        stack.push_back((*element).clone());
                    } else {
                        while let Some(ExpressionElement::Operator(_)) = stack.back() {
                            postfix.push(stack.pop_back().unwrap());
                        }
                        stack.pop_back();
                    }
                }
            }
        }

        while let Some(op) = stack.pop_back() {
            postfix.push(op);
        }

        Ok(Expression {
            expression: postfix,
            expression_type: Postfix,
        })
    }

    pub fn infix_to_prefix(&self) -> Result<Expression<Prefix>, ExpressionError> {
        let reversed_infix: Vec<ExpressionElement> = self.expression.iter().rev().map(|element| {
            match element {
                ExpressionElement::Parenthesis(c) => {
                    if *c == '(' {
                        ExpressionElement::Parenthesis(')')
                    } else {
                        ExpressionElement::Parenthesis('(')
                    }
                }
                _ => element.clone()
            }
        }).collect();

        let reversed_expression = Expression {
            expression: reversed_infix,
            expression_type: Infix,
        };
        println!("{}", reversed_expression);
        let postfix = reversed_expression.infix_to_postfix()?;
        println!("{}", postfix);
        let prefix: Vec<ExpressionElement> = postfix.expression.iter().rev().cloned().collect();

        Ok(Expression {
            expression: prefix,
            expression_type: Prefix,
        })
    }

}

impl Expression<Postfix> {
    pub fn postfix_to_infix(&self) -> Result<Expression<Infix>, ExpressionError> {
        let mut stack = Vec::new();

        for element in &self.expression {
            match element {
                ExpressionElement::Operand(_) => {
                    let operand = Expression {
                        expression: vec![element.clone()],
                        expression_type: Infix,
                    };
                    stack.push(operand);
                }
                ExpressionElement::Operator(op) => {
                    if stack.len() < 2 { //There must be 2 other elements to apply the operator to
                        return Err(ExpressionError::InvalidExpressionConversion);
                    }
                    let operand2 = stack.pop().unwrap();
                    let operand1 = stack.pop().unwrap();
                    let infix_expression = Expression {
                        expression: [
                            vec![ExpressionElement::Parenthesis('(')],
                            operand1.expression,
                            vec![ExpressionElement::Operator(op.clone())],
                            operand2.expression,
                            vec![ExpressionElement::Parenthesis(')')],
                        ].concat(),
                        expression_type: Infix,
                    };
                    stack.push(infix_expression);
                }
                _ => return Err(ExpressionError::InvalidExpressionConversion), //Since a postfix expression cannot contain parenthesis
            }
        }

        if stack.len() != 1 { //If the expression is not converted into a single complete infix expression
            return Err(ExpressionError::InvalidExpressionConversion);
        }

        Ok(stack.pop().unwrap())
    }

    pub fn postfix_to_prefix(&self) -> Result<Expression<Prefix>, ExpressionError> {
        let mut stack = Vec::new();

        for element in &self.expression {
            match element {
                ExpressionElement::Operand(_) => {
                    let operand = Expression {
                        expression: vec![element.clone()],
                        expression_type: Prefix,
                    };
                    stack.push(operand);
                }
                ExpressionElement::Operator(op) => {
                    if stack.len() < 2 {
                        return Err(ExpressionError::InvalidExpressionConversion);
                    }
                    let operand2 = stack.pop().unwrap();
                    let operand1 = stack.pop().unwrap();
                    let prefix_expression = Expression {
                        expression: [
                            vec![ExpressionElement::Operator(op.clone())],
                            operand1.expression,
                            operand2.expression
                        ].concat(),
                        expression_type: Prefix,
                    };
                    stack.push(prefix_expression);
                }
                _ => return Err(ExpressionError::InvalidExpressionConversion),
            }
        }

        if stack.len() != 1 {
            return Err(ExpressionError::InvalidExpressionConversion);
        }

        Ok(stack.pop().unwrap())
    }
}

impl Expression<Prefix> {
    pub fn prefix_to_infix(&self) -> Result<Expression<Infix>, ExpressionError> {
        let mut stack = Vec::new();

        for element in self.expression.iter().rev() {
            match element {
                ExpressionElement::Operand(_) => {
                    let operand = Expression {
                        expression: vec![element.clone()],
                        expression_type: Infix,
                    };
                    stack.push(operand);
                }
                ExpressionElement::Operator(op) => {
                    if stack.len() < 2 {
                        return Err(ExpressionError::InvalidExpressionConversion);
                    }
                    let operand1 = stack.pop().unwrap();
                    let operand2 = stack.pop().unwrap();
                    let infix_expression = Expression {
                        expression: [
                            vec![ExpressionElement::Parenthesis('(')],
                            operand1.expression,
                            vec![ExpressionElement::Operator(op.clone())],
                            operand2.expression,
                            vec![ExpressionElement::Parenthesis(')')]
                        ].concat(),
                        expression_type: Infix,
                    };
                    stack.push(infix_expression);
                }
                _ => return Err(ExpressionError::InvalidExpressionConversion)
            }
        }

        if stack.len() != 1 {
            return Err(ExpressionError::InvalidExpressionConversion);
        }

        Ok(stack.pop().unwrap())
    }

    pub fn prefix_to_postfix(&self) -> Result<Expression<Postfix>, ExpressionError> {
        let mut stack = Vec::new();

        for element in self.expression.iter().rev() {
            match element {
                ExpressionElement::Operand(_) => {
                    let operand = Expression {
                        expression: vec![element.clone()],
                        expression_type: Postfix,
                    };
                    stack.push(operand);
                }
                ExpressionElement::Operator(op) => {
                    if stack.len() < 2 {
                        return Err(ExpressionError::InvalidExpressionConversion);
                    }
                    let operand1 = stack.pop().unwrap();
                    let operand2 = stack.pop().unwrap();
                    let postfix_expression = Expression {
                        expression: [
                            operand1.expression,
                            operand2.expression,
                            vec![ExpressionElement::Operator(op.clone())]
                        ].concat(),
                        expression_type: Postfix,
                    };
                    stack.push(postfix_expression);
                }
                _ => return Err(ExpressionError::InvalidExpressionConversion)
            }
        }

        if stack.len() != 1 {
            return Err(ExpressionError::InvalidExpressionConversion);
        }

        Ok(stack.pop().unwrap())
    }
}

impl<T> TryFrom<(String, T)> for Expression<T> {
    type Error = ExpressionError;

    fn try_from(exp: (String, T)) -> Result<Self, Self::Error> {
        let mut expression = Vec::new();

        for c in exp.0.chars() {
            let element = match c {
                '+' | '-' | '*' | '/' => {
                    let operator = Operator::try_from(c)?;
                    ExpressionElement::Operator(operator)
                }
                '(' | ')' => ExpressionElement::Parenthesis(c),
                'a'..='z' => ExpressionElement::Operand(c),
                _ => return Err(ExpressionError::InvalidExpressionString),
            };

            expression.push(element);
        }

        Ok(Expression {
            expression,
            expression_type: exp.1,
        })
    }
}

impl<T> Display for Expression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut result = String::new();

        for element in &self.expression {
            match element {
                ExpressionElement::Operator(op) => result.push_str(&format!("{}", op)),
                ExpressionElement::Operand(c) | ExpressionElement::Parenthesis(c) => result.push(*c)
            }
        }

        write!(f, "{}", result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infix_to_postfix() {
        let expression = Expression::try_from(("a+b*c-d/e".to_string(), Infix)).unwrap();
        let converted = expression.infix_to_postfix().unwrap();
        assert_eq!(converted.to_string(), "abc*+de/-");
    }

    #[test]
    fn test_infix_to_prefix() {
        let expression = Expression::try_from(("a+b*c-d/e".to_string(), Infix)).unwrap();
        let converted = expression.infix_to_prefix().unwrap();
        assert_eq!(converted.to_string(), "+a-*bc/de");
    }

    #[test]
    fn test_postfix_to_infix() {
        let expression = Expression::try_from(("abc*+de/-".to_string(), Postfix)).unwrap();
        let converted = expression.postfix_to_infix().unwrap();
        assert_eq!(converted.to_string(), "((a+(b*c))-(d/e))");
    }

    #[test]
    fn test_postfix_to_prefix() {
        let expression = Expression::try_from(("abc*+de/-".to_string(), Postfix)).unwrap();
        let converted = expression.postfix_to_prefix().unwrap();
        assert_eq!(converted.to_string(), "-+a*bc/de");
    }

    #[test]
    fn test_prefix_to_infix() {
        let expression = Expression::try_from(("-+a*bc/de".to_string(), Prefix)).unwrap();
        let converted = expression.prefix_to_infix().unwrap();
        assert_eq!(converted.to_string(), "((a+(b*c))-(d/e))");
    }

    #[test]
    fn test_prefix_to_postfix() {
        let expression = Expression::try_from(("-+a*bc/de".to_string(), Prefix)).unwrap();
        let converted = expression.prefix_to_postfix().unwrap();
        assert_eq!(converted.to_string(), "abc*+de/-");
    }
}
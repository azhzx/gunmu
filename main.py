from gunmu import parse


EXAMPLE = '''\
module demo

record Point {
    x: Int
    y: Int
}

theorem add_zero(x: Int): x + 0 == x
proof {
        simp
    }

fn main(value: Int) -> Unit
requires value >= 0
ensures value >= 0 {
    let point = Point { x: value, y: 1 }
    print point.x

    match value {
        0 => {
            print "zero"
        }
        n if n > 0 => {
            print "positive"
        }
        _ => {
            print "negative"
        }
    }

    loop (i = 0, sum = 0)
    with {
        invariant i >= 0
        invariant sum >= 0
        variant 10 - i
    } {
        match i {
            10 => {
                break sum
            }
            _ => {
                continue (i = i + 1, sum = sum + i)
            }
        }
    }

    print value
    return unit
}
'''


if __name__ == "__main__":
    import pprint
    pprint.pp(parse(EXAMPLE))

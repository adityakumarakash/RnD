Definition divide (x y : N) := exists z, x * z = y.
Definition prime x := forall y, divide y x -> y = 1 \/ y = x.

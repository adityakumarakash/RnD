Structure monoid := { 
  dom : Type ; 
  op : dom -> dom -> dom where "x * y" := (op x y); 
  id : dom where "1" := id; 
  assoc : ∀ x y z, x * (y * z) = (x * y) * z ; 
  left_neutral : ∀ x, 1 * x = x ;
  right_neutral : ∀ x, x * 1 = x 
}.

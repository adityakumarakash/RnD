length_corr =
  fun (n : nat) (s : seq n) =>
    seq_ind (fun (n0 : nat) (s0 : seq n0) => length n0 s0 = n0) 
      (refl_equal 0)
      (fun (n0 _ : nat) (s0 : seq n0) (IHs : length n0 s0 = n0) =>
        eq_ind_r 
          (fun n2 : nat => S n2 = S n0) 
          (refl_equal (S n0)) IHs) n s
: forall (n : nat) (s : seq n), length n s = n.

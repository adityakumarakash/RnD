Theorem plus_assoc : forall n m p : nat,
  n + (m + p) = (n + m) + p.
Proof.
  intros n m p.
  induction n as [|n'].
  Case "0". simpl. reflexivity.
  Case "S m". simpl. rewrite -> IHn'. reflexivity.
Qed.

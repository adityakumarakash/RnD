Inductive nat : Set :=
  | 0 : nat
  | S : nat -> nat.

Inductive list (A:Type) : Type :=
  | nil : list A
  | cons : A -> list A -> list A.

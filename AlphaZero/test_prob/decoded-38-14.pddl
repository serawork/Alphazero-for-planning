(define (problem BLOCKS-genProblems)
(:domain BLOCKS)
(:objects A B C D E F G H I J K L M N - block)
(:INIT (CLEAR B) (CLEAR C) (CLEAR F) (CLEAR G) (CLEAR H) (CLEAR J) (CLEAR L) (CLEAR N) (HANDEMPTY) (ONTABLE D) (ONTABLE G) (ONTABLE H) (ONTABLE I) (ONTABLE K) (ONTABLE L) (ONTABLE M) (ONTABLE N) (ON A I) (ON B D) (ON C A) (ON E K) (ON F M) (ON J E))
(:goal (AND (ON H G) (ON N L) (ON A I) (ON B D) (ON C A) (ON E K) (ON F M) (ON J E)))
)

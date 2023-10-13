test = {
  'name': 'make-list',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          scm> (define a '(1))
          a
          scm> a
          (1)
          scm> (define b (cons 2 a))
          b
          scm> b
          (2 1)
          scm> (define c (list 3 b))
          c
          scm> c
          (3 (2 1))
          scm> (car c)
          3
          scm> (cdr c)
          ((2 1))
          scm> (car (car (cdr c)))
          8f01429f05539100445ff1214be81240
          # locked
          scm> (cdr (car (cdr c)))
          2c988a84918e0b2958168830ef8b7391
          # locked
          """,
          'hidden': False,
          'locked': True
        },
        {
          'code': r"""
          scm> lst  ; type out exactly how Scheme would print the list
          d27fa7ae58e6e5c9334663d5f70ed8d8
          # locked
          """,
          'hidden': False,
          'locked': True
        }
      ],
      'scored': True,
      'setup': r"""
      scm> (load-all ".")
      """,
      'teardown': '',
      'type': 'scheme'
    }
  ]
}

import game_generator

sample_template = """
Yesterday,
 ____ (person) brought a
 ____ (item) and used it to
 ____ (verb) a
 ____ (object)
"""

result = game_generator.GameGenerator().run(sample_template)
print(f"Input: {sample_template}")
print(f"Output: {result}")

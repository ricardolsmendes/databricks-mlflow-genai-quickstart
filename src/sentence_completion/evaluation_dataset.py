import message_factory

factory_fn = message_factory.SentenceCompletionMessageFactory.make_input_messages
eval_data = [
    {
        "inputs": {
            "messages": factory_fn(
                "Yesterday, ____ (person) brought a ____ (item) and used it to"
                " ____ (verb) a ____ (object)"
            )
        }
    },
    {
        "inputs": {
            "messages": factory_fn(
                "I wanted to ____ (verb) but ____ (person) told me to"
                " ____ (verb) instead"
            )
        }
    },
    {
        "inputs": {
            "messages": factory_fn(
                "The ____ (adjective) ____ (animal) likes to ____ (verb) in the"
                " ____ (place)"
            )
        }
    },
    {
        "inputs": {
            "messages": factory_fn(
                "My favorite ____ (food) is made with ____ (ingredient) and"
                " ____ (ingredient)"
            )
        }
    },
    {
        "inputs": {
            "messages": factory_fn(
                "When I grow up, I want to be a ____ (job) who can ____ (verb) all day"
            )
        }
    },
    {
        "inputs": {
            "messages": factory_fn(
                "When two ____ (animals) love each other, they ____ (verb) under the"
                " ____ (place)"
            )
        }
    },
    {
        "inputs": {
            "messages": factory_fn(
                "The monster wanted to ____ (verb) all the ____ (plural noun) with its"
                " ____ (body part)"
            )
        }
    },
]

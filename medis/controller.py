""" Functions that are common to each can be applied to each object """


def auto_load_single(obj_type):
    """
    load or generate the data for a single object

    :param obj_type:
    :return:
    """

    obj = obj_type()
    if obj.can_load():
        obj.load()
    else:
        obj.generate()
        if obj.use_cache:
            obj.save()

    if obj.debug:
        obj.view()

    return obj

def auto_load(obj_types):
    """
    Load/generate data for an object or list of objects

    >>> Coronagraph
    >>> [Coronagraph]

    obj_types list()

    :return:
    """

    if isinstance(obj_types, list):
        objs = []  # the plural of the plural
        for obj_type in obj_types:
            obj = auto_load_single(obj_type)
            objs.append(obj)
    else:
        objs = auto_load_single(obj_types)

    return objs
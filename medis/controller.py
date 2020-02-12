""" Functions that are common to each can be applied to each object """


def get_obj_data(obj_type):
    """
    load or generate the data for a single object

    :param obj_type:
    :return:
    """

    obj = obj_type()
    if obj.can_load():
        obj.data = obj.load()
    else:
        obj.data = obj.generate()
        if sp.usecache:
            obj.save()

    if sp.debug:
        obj.view()

    return obj.data

def get_data(obj_types):
    """
    Load/generate data for an object or list of objects

    >>> Coronagraph
    >>> [Coronagraph]

    obj_types list()

    :return:
    """

    if isinstance(obj_types, list):
        datas = []  # the plural of the plural
        for obj_type in obj_types:
            data = get_obj_data(obj_type)
            datas.append(data)
    else:
        datas = get_obj_data(obj_types)

    return datas
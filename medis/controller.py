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

def configs_match(obj):
    cur_config = obj.__dict__
    cache_config = np.loadtxt(obj.cache_name)
    configs_match = cur_config == cache_config

    return configs_match

def can_load(obj):
    if sp.use_cache:
        file_exists = os.path.exists(obj.cache_name)
        if file_exists:
            configs_match = configs_match(obj)
            if configs_match:
                return True

    return False
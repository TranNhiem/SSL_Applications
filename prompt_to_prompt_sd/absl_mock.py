from six import string_types  # compitible with py2x, py3x str object

class Mock_Flag(object):
    _singleton_inst = None
    # __new__ is used to create the class before calling __init__,
    # so we set the singleton guard in here for implement the singleton design pattern
    def __new__(cls, *args, **kwargs): 
        # new a instance, if no instance have been created 
        if cls._singleton_inst is None:     
            # store 'the' instance into staitc member
            cls._singleton_inst = super().__new__(cls) 
            # create the 'common' flag space allow all instance access
            cls.flag_spec = type("FLAG_spec", (), {})()
        # next time create the other instance, 
        # just let it point to 'the' singleton instance!
        return cls._singleton_inst
  
    @property
    def FLAGS(self):
        return self.flag_spec

    def DEFINE_string(self, var_name, value, helper_str):
        if isinstance(value, string_types) or value == None:
            self.flag_spec.__dict__[var_name] = value
        else:
            raise TypeError("The input value should be string type")
    
    def DEFINE_boolean(self, var_name, value, helper_str):
        if isinstance(value, bool) or value == None:
            self.flag_spec.__dict__[var_name] = value
        else:
            raise TypeError("The input value should be boolean type")

    def DEFINE_integer(self, var_name, value, helper_str):
        if isinstance(value, int) or value == None:
            self.flag_spec.__dict__[var_name] = value
        else:
            raise TypeError("The input value should be interger type")
    
    def DEFINE_float(self, var_name, value, helper_str):
        if isinstance(value, float) or isinstance(value, int) or value == None:
            self.flag_spec.__dict__[var_name] = value
        else:
            raise TypeError("The input value should be float type")
    
    def DEFINE_enum(self, var_name, value, choice_lst, helper_str):
        assert len(choice_lst) > 0, "The enum type should give the choice list"
        if value in choice_lst:
            self.flag_spec.__dict__[var_name] = value
        else:
            raise TypeError("The input value should be belong into the choice list")

    def DEFINE_dict(self, var_name, value, helper_str):
        if isinstance(value, dict) or value == None:
            self.flag_spec.__dict__[var_name] = value
        else:
            raise TypeError("The input value should be dict type")

    def save_config(self,file_name):
        with open(file_name,"w") as f:
            for key in self.flag_spec.__dict__.keys():
                f.write(key + " : " +str(self.flag_spec.__dict__[key])+"\n")
        print("save the config in :",file_name)
    

def local_test():
    flag_inst = Mock_Flag()
    FLAGS = flag_inst.FLAGS

    flag_inst.DEFINE_integer("age", 18, "no way!!")
    flag_inst.DEFINE_float("lucky_num", 13.35, "This is my lucky number, you may got one")
    flag_inst.DEFINE_boolean("lawyer", True, "lawyer said the True ~ ~")
    flag_inst.DEFINE_string("name", "no_name", "just kidding@@")
    flag_inst.DEFINE_enum("car", "tesla", ("honda", "tesla", "foxconn"), "foxconn not bad..")
    flag_inst.DEFINE_dict("book", {"joseph":32.5, "rick":False}, "ok")

    print(f"my ages : {FLAGS.age}\n")
    print(f"my lucky_num : {FLAGS.lucky_num}\n")
    print(f"I have lawyer ? {FLAGS.lawyer}\n")
    print(f"my name : {FLAGS.name}\n")
    print(f"my car : {FLAGS.car}\n")
    print(f"rick dict : {FLAGS.book['rick']}\n")

    cp_flg = Mock_Flag()
    FLAGS_CP = cp_flg.FLAGS
    cp_flg.DEFINE_integer("score", 99, "no way!!")
    print(f"test name : {FLAGS_CP.name}\n")
    print(f"old args : {FLAGS.score}\n")

# you can run the local test in here
if __name__ == '__main__':
    local_test()
    # simulate the out of space 
    flag_inst = Mock_Flag()
    FLAGS = flag_inst.FLAGS
    print(f"test name : {FLAGS.name} done!!\n")
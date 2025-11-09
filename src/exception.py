#VERY IMPORTANT -> When exception occurs, the sys command will have the access and the control.

import sys
from logger import logging

#xe_tb will provide all the information about the errors

#Why use _, -> The function returns some value and we are ignoring that value, we can worry about ->exc_tb

def error_message_detail(error, error_detail:sys): 
    _, _, exc_tb=error_detail.exc_info()   #This returns tuple [type, value and traceback , exc_tb]
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message='Error occured in the python script name[{0}] line number [{1}] error message [{2}]'.format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
# if __name__=='__main__':
#     try:
#        a=1/0
#     except Exception as e:
#        logging.info('Divide by zero')
#        raise CustomException(e, sys)

# The Below is to check this file working and save to the log file
# python src/exception.py

#NOTES: src.logger -> Some error -> We should import directly as logger


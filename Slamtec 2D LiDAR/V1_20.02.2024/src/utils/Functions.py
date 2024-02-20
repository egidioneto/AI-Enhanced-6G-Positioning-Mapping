import time
import psutil


# ---------------------------------------------------------------------------- #
#                        Util file with useful functions                       #
# ---------------------------------------------------------------------------- #

def Convert_list_to_dic(lst, lst2):
    """This method turns two lists into one dictionary"""
    return {lst[i]: lst2[i] for i in range(len(lst))}

def get_cpu_usage():
    """
    CPU: Look at the Change in CPU Usage value. If it's high (e.g., above 70%), you'll 
    need a Raspberry Pi with a faster CPU. Raspberry Pi models vary in CPU speed and cores.
    """
    return psutil.cpu_percent(interval=1)

def get_ram_usage():
    """
    RAM: The Change in RAM Usage value will indicate how much additional RAM your script requires. 
    Add this to the base RAM usage of the operating system and any other running services. Raspberry 
    Pi models offer different RAM sizes (e.g., 2GB, 4GB, 8GB).
    """

    process = psutil.Process()
    ram_in_bytes = process.memory_info().rss
    ram_in_mb = ram_in_bytes / (1024 ** 2)
    return ram_in_mb

def get_network_usage(duration=1):
    """
    The duration parameter in the get_network_usage function specifies the time interval in seconds 
    over which the network usage is measured. The function captures the network I/O counters at the 
    start and end of this interval, and then calculates the difference to determine the bytes sent 
    and received during that time.
    """

    """
    Network: If the Change in Network Usage values for sent and received data are high, 
    ensure that the Raspberry Pi model you choose has adequate network capabilities. Some 
    models offer Gigabit Ethernet and better Wi-Fi support.
    """

    net_start = psutil.net_io_counters()
    time.sleep(duration)
    net_end = psutil.net_io_counters()

    sent_bytes = net_end.bytes_sent - net_start.bytes_sent
    received_bytes = net_end.bytes_recv - net_start.bytes_recv

    sent_mbps = (sent_bytes * 8) / (1024 ** 2) / duration
    received_mbps = (received_bytes * 8) / (1024 ** 2) / duration

    return received_mbps, sent_mbps

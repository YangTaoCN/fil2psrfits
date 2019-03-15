import numpy as np
from astropy.io import fits
import astropy.time as atime
import astropy.coordinates as acoord
import astropy.units as aunits
import fill_headers as fillhdr
import sigproc 
import os, glob

class ObsInfo(object):
    def __init__(self):
        self.file_date = self.format_date(atime.Time.now().isot)
        self.observer  = "RSW"
        self.proj_id   = "FRB FASTVIS"
        self.obs_date  = ""
        self.fcenter   = 0.0
        self.bw        = 0.0
        self.nchan     = 0
        self.src_name  = ""
        self.ra_str    =  "00:00:00"
        self.dec_str   = "+00:00:00"
        self.bmaj_deg  = 0.0
        self.bmin_deg  = 0.0
        self.bpa_deg   = 0.0
        self.scan_len  = 0
        self.stt_imjd  = 0
        self.stt_smjd  = 0
        self.stt_offs  = 0.0
        self.stt_lst   = 0.0        

        self.dt        = 0.0
        self.nbits     = 32
        self.nsuboffs  = 0.0
        self.chan_bw   = 0.0
        self.nsblk     = 0

        # Some more stuff you prob don't need to change
        self.telescope = 'VLA'
        self.ant_x     = -1601185.63
        self.ant_y     = -5041978.15
        self.ant_z     =  3554876.43
        self.longitude = self.calc_longitude()

    def calc_longitude(self):
        cc = acoord.EarthLocation.from_geocentric(self.ant_x,
                                                  self.ant_y,
                                                  self.ant_z,
                                                  unit='m')
        longitude = cc.longitude.deg
        return longitude
        
    def fill_from_mjd(self, mjd):
        stt_imjd = int(mjd)
        stt_smjd = int((mjd - stt_imjd) * 24 * 3600)
        stt_offs = ((mjd - stt_imjd) * 24 * 3600.0) - stt_smjd
        self.stt_imjd = stt_imjd
        self.stt_smjd = stt_smjd
        self.stt_offs = stt_offs
        self.obs_date = self.format_date(atime.Time(mjd, format='mjd').isot)
        
    def fill_freq_info(self, fcenter, nchan, chan_bw):
        self.fcenter = fcenter
        self.bw      = np.abs(nchan * chan_bw)
        self.nchan   = nchan
        self.chan_bw = chan_bw

    def fill_source_info(self, src_name, ra_str, dec_str):
        self.src_name = src_name
        self.ra_str   = ra_str
        self.dec_str  = dec_str

    def fill_beam_info(self, bmaj_deg, bmin_deg, bpa_deg):
        self.bmaj_deg = bmaj_deg
        self.bmin_deg = bmin_deg
        self.bpa_deg  = bpa_deg

    def fill_data_info(self, dt, nbits):
        self.dt = dt
        self.nbits = nbits

    def calc_start_lst(self, mjd):
        self.stt_lst = self.calc_lst(mjd, self.longitude)

    def calc_lst(self, mjd, longitude):
        gfac0 = 6.697374558
        gfac1 = 0.06570982441908
        gfac2 = 1.00273790935
        gfac3 = 0.000026
        mjd0 = 51544.5               # MJD at 2000 Jan 01 12h
        H  = (mjd - int(mjd)) * 24   # Hours since previous 0h
        D  = mjd - mjd0              # Days since MJD0
        D0 = int(mjd) - mjd0         # Days between MJD0 and prev 0h
        T  = D / 36525.0             # Number of centuries since MJD0
        gmst = gfac0 + gfac1 * D0 + gfac2 * H + gfac3 * T**2.0
        lst = ((gmst + longitude/15.0) % 24.0) * 3600.0
        return lst

    def format_date(self, date_str):
        # Strip out the decimal seconds
        out_str = date_str.split('.')[0]
        return out_str
        


def calc_lst(mjd, longitude):
    gfac0 = 6.697374558
    gfac1 = 0.06570982441908
    gfac2 = 1.00273790935
    gfac3 = 0.000026
    mjd0 = 51544.5               # MJD at 2000 Jan 01 12h
    H  = (mjd - int(mjd)) * 24   # Hours since previous 0h
    D  = mjd - mjd0              # Days since MJD0
    D0 = int(mjd) - mjd0         # Days between MJD0 and prev 0h
    T  = D / 36525.0             # Number of centuries since MJD0
    gmst = gfac0 + gfac1 * D0 + gfac2 * H + gfac3 * T**2.0
    lst = ((gmst + longitude/15.0) % 24.0) * 3600.0
    return lst


def get_deg_coords_from_ra_dec(ra_str, dec_str, coordsys='fk5'):
    c = acoord.SkyCoord(ra_str, dec_str, frame='icrs', 
                        unit=(aunits.hourangle, aunits.deg))
    if coordsys=='fk5':
        return c.ra.deg, c.dec.deg
    elif coordsys=='galactic':
        return c.galactic.l.deg, c.galactic.b.deg
    else:
        print("Invalid coordsys")
        return


def initialize_psrfits(outname, nsamps, nchans, nifs, nbits, npsub=-1,  
	               src_name="", ra_str="00:00:00", dec_str="00:00:00",
                       mjd_start=0.0, dt=0.001, freq_lo=1400.0, chan_df=1.0,
                       beam_info=np.array([0.0, 0.0, 0.0])):
    """
    Set up a PSRFITS file with everything set up EXCEPT 
    the DATA.  We want to be able to add the data in later	
    """
    # Obs Specific Metadata
    # Time Info
    mjd      = mjd_start
    dt       = dt  # seconds
    print("  MJD START: %.8f  " %mjd_start)
    print("  TIME RES : %.3f ms" %(dt * 1000.0))
    # Frequency Info (All freqs in MHz)
    chan_bw  = chan_df
    bw       = nchans * chan_bw
    fcenter  = freq_lo + 0.5 * (nchans - 1) * chan_df
    freqs    = np.arange(nchans) * chan_df + freq_lo 
    print("  nchans   : %d" %nchans)
    print("  chan_df  : %.2f MHz" %chan_bw)
    print("  fcenter  : %.2f MHz" %fcenter)
    # Source Info
    src_name = src_name
    ra_str   = ra_str
    dec_str  = dec_str
    print("  SOURCE NAME : %s" %src_name)
    print("  SOURCE RA   : %s" %ra_str)
    print("  SOURCE DEC  : %s" %dec_str)
    # Beam Info
    bmaj_deg = beam_info[0] / 3600.0 
    bmin_deg = beam_info[1] / 3600.0
    bpa_deg  = beam_info[2]
    print("  BMAJ : %.1f arcsec" %(bmaj_deg * 3600.0))
    print("  BMIN : %.1f arcsec" %(bmin_deg * 3600.0))
    print("  BPA  : %.1f deg" %bpa_deg)
    # Output file name
    outfile = "%s.fits" %outname
    print(" OUTPUT FILE NAME: %s" %outfile)
    
    # Fill in the ObsInfo class
    d = ObsInfo()
    d.fill_from_mjd(mjd)
    d.fill_freq_info(fcenter, nchans, chan_bw)
    d.fill_source_info(src_name, ra_str, dec_str)
    d.fill_beam_info(bmaj_deg, bmin_deg, bpa_deg)
    d.fill_data_info(dt, nbits)
    d.calc_start_lst(mjd)

    # Determine subint size for PSRFITS table
    if npsub > 0:
       n_per_subint = npsub
    else:
       n_per_subint = int(1.0 / dt)
    n_subints    = int(nsamps / n_per_subint)
    t_subint     = n_per_subint * dt
    d.nsblk    = n_per_subint
    d.scan_len = t_subint * n_subints

    # Reshape data array
    #data = data[: n_per_subint * n_subints]
    #data = data.reshape( (n_subints, n_per_subint * nchans) )
    data = np.array([])
   
    tstart = 0.0 
    # Prepare arrays for columns
    tsubint  = np.ones(n_subints, dtype=np.float64) * t_subint
    offs_sub = (np.arange(n_subints) + 0.5) * t_subint + tstart
    lst_sub  = np.array([ calc_lst(mjd + tsub / (24. * 3600.0), d.longitude) \
                              for tsub in offs_sub ], dtype=np.float64)
    ra_deg, dec_deg = get_deg_coords_from_ra_dec(ra_str, dec_str, coordsys='fk5')
    l_deg, b_deg    = get_deg_coords_from_ra_dec(ra_str, dec_str, coordsys='galactic')
    ra_sub   = np.ones(n_subints, dtype=np.float64) * ra_deg 
    dec_sub  = np.ones(n_subints, dtype=np.float64) * dec_deg
    glon_sub = np.ones(n_subints, dtype=np.float64) * l_deg
    glat_sub = np.ones(n_subints, dtype=np.float64) * b_deg
    fd_ang   = np.zeros(n_subints, dtype=np.float32)
    pos_ang  = np.zeros(n_subints, dtype=np.float32)
    par_ang  = np.zeros(n_subints, dtype=np.float32)
    tel_az   = np.zeros(n_subints, dtype=np.float32)
    tel_zen  = np.zeros(n_subints, dtype=np.float32)
    dat_freq = np.vstack( [freqs] * n_subints ).astype(np.float32)

    dat_wts  = np.ones(  (n_subints, nchans), dtype=np.float32 )
    dat_offs = np.zeros( (n_subints, nchans), dtype=np.float32 )
    dat_scl  = np.ones(  (n_subints, nchans), dtype=np.float32 )

    # Make the columns
    tbl_columns = [
        fits.Column(name="TSUBINT" , format='1D', unit='s', array=tsubint),
        fits.Column(name="OFFS_SUB", format='1D', unit='s', array=offs_sub),
        fits.Column(name="LST_SUB" , format='1D', unit='s', array=lst_sub),
        fits.Column(name="RA_SUB"  , format='1D', unit='deg', array=ra_sub),
        fits.Column(name="DEC_SUB" , format='1D', unit='deg', array=dec_sub),
        fits.Column(name="GLON_SUB", format='1D', unit='deg', array=glon_sub),
        fits.Column(name="GLAT_SUB", format='1D', unit='deg', array=glat_sub),
        fits.Column(name="FD_ANG"  , format='1E', unit='deg', array=fd_ang),
        fits.Column(name="POS_ANG" , format='1E', unit='deg', array=pos_ang),
        fits.Column(name="PAR_ANG" , format='1E', unit='deg', array=par_ang),
        fits.Column(name="TEL_AZ"  , format='1E', unit='deg', array=tel_az),
        fits.Column(name="TEL_ZEN" , format='1E', unit='deg', array=tel_zen),
        fits.Column(name="DAT_FREQ", format='%dE'%nchans, unit='MHz', array=dat_freq),
        fits.Column(name="DAT_WTS" , format='%dE'%nchans, array=dat_wts),
        fits.Column(name="DAT_OFFS", format='%dE'%nchans, array=dat_offs),
        fits.Column(name="DAT_SCL" , format='%dE'%nchans, array=dat_scl),
        fits.Column(name="DATA"    , format=str(nifs*nchans*n_per_subint*nbits/8) + 'B', 
                    dim='(%d,%d,%d)' %(nchans, nifs, n_per_subint), array=data),
    ]

    # Fill in the headers
    phdr = fillhdr.fill_primary_header(d)
    thdr = fillhdr.fill_table_header(d)
    fits_data = fits.HDUList()

    # Add the columns to the table
    print(" Building the PSRFITS table")
    table_hdu = fits.BinTableHDU(fits.FITS_rec.from_columns(tbl_columns), 
                                 name="subint", header=thdr)

    # Add primary header
    primary_hdu = fits.PrimaryHDU(header=phdr)

    # Add hdus to FITS file and write 
    print(" Writing file...")
    fits_data.append(primary_hdu)
    fits_data.append(table_hdu)
    fits_data.writeto(outfile, clobber=True)
    print(" Done.")

    return


def convert_fil2psrfits(filfile, outname, npsub=-1):
    """
    Convert one filterbank file to one psrfits file
    """
    # Read in header info
    print("Reading header from %s" %(filfile))
    fmtdict = sigproc.fmtdict
    hd = sigproc.get_header_dict(filfile, 4, fmtdict)

    nsamps = hd['nspec']
    nchans = hd['nchans']
    nifs   = hd['nifs']
    nbits  = hd['nbits']
    mjd_start = hd['tstart']
    dt     = hd['tsamp']
    fch1   = hd['fch1']
    foff   = hd['foff']
    hdr_size = hd['hsize'] 

    freqs = fch1 + np.arange(nchans) * foff
    df = np.abs(foff)
    freq_lo = np.min(freqs)

    # If npsub is not give, pick something so subint is 
    # close to a second 
    if npsub < 0:
       npsub = int(1.0 / dt)
    else:
       pass
   
    # Create and initialize the PSRFITS file 
    print("Initializing PSRFITS file")
    initialize_psrfits(outname, nsamps, nchans, nifs, nbits, 
	               npsub=npsub, src_name=hd['src_name'], 
                       ra_str=hd['src_raj'], dec_str=hd['src_dej'],
                       mjd_start=hd['tstart'], dt=dt, 
                       freq_lo=freq_lo, chan_df=df)
    
    # Fill the new PSRFITS file with data 
    print("Filling PSRFITS file with filterbank data")

    # Open PSRFITS file 
    fitsfile = "%s.fits" %(outname)
    hdulist = fits.open(fitsfile, mode='update')
    hdu = hdulist[1]
    Nsub = len(hdu.data[:]['data'])
    
    # Open fil file and skip header
    f = open(filfile)
    f.seek(hdr_size, os.SEEK_SET)  

    # Loop through chunks of data to write to PSRFITS
    n_read_subints = 10
    for istart in np.arange(0, Nsub, n_read_subints):
        print("%d/%d" %(istart, Nsub))
        istop = istart + n_read_subints
        if istop > Nsub:
            istop = Nsub
        else:
            pass
        isub = istop - istart  
        
        # Read in nread samples from filfile
        nread = isub * nchans * nifs * npsub 
        dd = np.fromfile(f, dtype='uint8', count=nread)
        dd = np.reshape(dd, (isub, npsub, nifs, nchans))

        # Put data in hdu data array
        hdu.data[istart : istop]['data'] = dd[:]

        # Write to file
        hdulist.flush()

    # Close open files
    hdulist.close()
    f.close()
 
    return 


def get_min_max_freqs(hd):
    fch1 = hd['fch1']
    foff = hd['foff']
    nchans = hd['nchans']
    if foff > 0:
        fmin = fch1
        fmax = fch1 + (nchans-1) * foff
    elif foff < 0:
        fmin = fch1 + (nchans-1) * foff
        fmax = fch1 
    else:
	print("foff = 0 --> SOMETHING HAS GONE WRONG")
        fmin = fch1
        fmax = fch1
    return fmin, fmax


def check_headers_and_get_info(hd_list):
    hd_out = {}

    # -- consistency checks -- 
    err_code = 0

    # nsamps
    nsamps_arr = np.array([ hd['nspec'] for hd in hd_list ])
    if len(np.unique(nsamps_arr)) > 1:
        print("Multiple nsamps values found!")
        err_code += 10**0
    else: 
	hd_out['nspec'] = np.unique(nsamps_arr)[0]
        

    # nifs
    nifs_arr = np.array([ hd['nifs'] for hd in hd_list ])
    if len(np.unique(nifs_arr)) > 1:
        print("Multiple nifs values found!")
        err_code += 10**1
    else: 
	hd_out['nifs'] = np.unique(nifs_arr)[0]

    # nbits
    nbits_arr = np.array([ hd['nbits'] for hd in hd_list ])
    if len(np.unique(nbits_arr)) > 1:
        print("Multiple nbits values found!")
        err_code += 10**2
    else: 
	hd_out['nbits'] = np.unique(nbits_arr)[0]

    # mjd_starts
    mjd_arr = np.array([ hd['tstart'] for hd in hd_list ])
    if len(np.unique(mjd_arr)) > 1:
        print("Multiple mjd_start values found!")
        err_code += 10**3
    else: 
	hd_out['tstart'] = np.unique(mjd_arr)[0]

    # time resolution
    dt_arr = np.array([ hd['tsamp'] for hd in hd_list ])
    if len(np.unique(dt_arr)) > 1:
        print("Multiple sample time values found!")
        err_code += 10**4
    else: 
	hd_out['tsamp'] = np.unique(dt_arr)[0]

    # freq resolution
    df_arr = np.array([ np.abs(hd['foff']) for hd in hd_list ])
    if len(np.unique(df_arr)) > 1:
        print("Multiple channel width values found!")
        err_code += 10**5
    else: 
	hd_out['df'] = np.unique(df_arr)[0]

    # Source Name
    src_names = np.array([ hd['src_name'] for hd in hd_list ])
    if len(np.unique(src_names)) > 1:
        print("Multiple Source Names found!")
        err_code += 10**6
    else: 
	hd_out['src_name'] = np.unique(src_names)[0]

    # DEC STRING
    dec_vals = np.array([ hd['src_dej'] for hd in hd_list ])
    if len(np.unique(dec_vals)) > 1:
        print("Multiple Source DEC values found!")
        err_code += 10**7
    else: 
	hd_out['src_dej'] = np.unique(dec_vals)[0]
    
    # RA STRING
    ra_vals = np.array([ hd['src_raj'] for hd in hd_list ])
    if len(np.unique(ra_vals)) > 1:
        print("Multiple Source RA values found!")
        err_code += 10**8
    else: 
	hd_out['src_raj'] = np.unique(ra_vals)[0]

    # min / max freqs
    min_freqs = []
    max_freqs = []
    for hd in hd_list:
        fmin, fmax = get_min_max_freqs(hd)
        min_freqs.append(fmin)
        max_freqs.append(fmax)
    
    hd_out['fmin'] = np.min(min_freqs)
    hd_out['fmax'] = np.max(max_freqs) 

    # Get fch1 values  
    hd_out['fch1s'] = np.array([ hd['fch1'] for hd in hd_list ])
  
    # Get foff values
    hd_out['foffs'] = np.array([ hd['foff'] for hd in hd_list ])

    # Get nchan values
    hd_out['nchans'] = np.array([ hd['nchans'] for hd in hd_list ])

    # Get hdr_sizes 
    hd_out['hdr_sizes'] = np.array([ hd['hsize'] for hd in hd_list ])

    return hd_out, err_code  


def combine_convert_fil2psrfits(filfiles, outname, npsub=-1, maxpol=4):
    """
    Convert one filterbank file to one psrfits file
    """
    # Read in header info
    fmtdict = sigproc.fmtdict
    hd_list = []
    for filfile in filfiles:
        print("Reading header from %s" %(filfile))
        hdi = sigproc.get_header_dict(filfile, 4, fmtdict)
	hd_list.append(hdi)

    hd, err_code = check_headers_and_get_info(hd_list)

    if err_code > 0:
        print("ERROR CODE: %d" %err_code)
        return
    else: pass

    # Single Values
    nsamps    = hd['nspec']
    nifs      = hd['nifs']
    nbits     = hd['nbits']
    mjd_start = hd['tstart']
    dt        = hd['tsamp']
    df        = hd['df']
    freq_lo   = hd['fmin']
    freq_hi   = hd['fmax']
    src_name  = hd['src_name']
    src_raj   = hd['src_raj']
    src_dej   = hd['src_dej']

    # Arrays
    fch1s   = hd['fch1s']
    foffs   = hd['foffs']
    nchan_vals  = hd['nchans']
    hdr_sizes   = hd['hdr_sizes'] 

    nchans = int( (freq_hi - freq_lo) / df ) + 1
    freqs = freq_lo + np.arange(nchans) * df 

    # Set nifs_out
    if nifs > maxpol:
       nifs_out = maxpol
       print("ONLY USING %d POLARIZATIONS" %maxpol)
    else:
       nifs_out = nifs

    # If npsub is not given, pick something so subint is 
    # close to a second 
    if npsub < 0:
       npsub = int(1.0 / dt)
    else:
       pass

    print("nchans = %d" %nchans)
    print("nsamps = %d" %nsamps)
    print("FREQ LO = %f" %freq_lo)
    print("FREQ HI = %f" %freq_hi)
  
    #""" 
    # Create and initialize the PSRFITS file 
    print("Initializing PSRFITS file")
    initialize_psrfits(outname, nsamps, nchans, nifs_out, nbits, 
	               npsub=npsub, src_name=src_name, 
                       ra_str=src_raj, dec_str=src_dej, 
                       mjd_start=mjd_start, dt=dt, 
                       freq_lo=freq_lo, chan_df=df)
    #"""

    # Fill the new PSRFITS file with data 
    print("Filling PSRFITS file with filterbank data")

    # We'll make multiple passes to fill in the FITS 
    # file for each fil file.

    # Open PSRFITS file 
    fitsfile = "%s.fits" %(outname)
    hdulist = fits.open(fitsfile, mode='update')
    hdu = hdulist[1]
    Nsub = len(hdu.data[:]['data'])

    for ii, filfile in enumerate(filfiles):
	print("Filling in data for %s" %filfile)

        # Set up some info for this filfile
        hdr_size_ii = hdr_sizes[ii]
        fmin, fmax = get_min_max_freqs(hd_list[ii]) 
        nchans_ii = nchan_vals[ii]
        foff_ii  = foffs[ii]
        f_start = int( (fmin - freqs[0]) / df )
	f_stop = f_start + nchans_ii

        print(" CURRENT FIL INFO")
        print("   freq_lo = %f" %(fmin))
        print("   freq_hi = %f" %(fmax))
        print(" ")
        print("   flo_bin = %d" %(f_start))
        print("   fhi_bin = %d" %(f_stop)) 
    
    	# Open fil file and skip header
    	f = open(filfile)
    	f.seek(hdr_size_ii, os.SEEK_SET)  
    	
	# Loop through chunks of data to write to PSRFITS
    	n_read_subints = 10
    	for istart in np.arange(0, Nsub, n_read_subints):
            print("(%d/%d) -- %d/%d" %(ii+1, len(filfiles), istart, Nsub))
            istop = istart + n_read_subints
            if istop > Nsub:
                istop = Nsub
            else:
                pass
            isub = istop - istart  
            
            # Read in nread samples from filfile
            nread = isub * nchans_ii * nifs * npsub 
            dd = np.fromfile(f, dtype='uint8', count=nread)
            dd = np.reshape(dd, (isub, npsub, nifs, nchans_ii))

	    dd = dd[:, :, :nifs_out, :]
	    
            # If foff is negative, we need to flip the freq axis
            if foff_ii < 0:
                #print("Flipping band")
                dd = dd[:, :, :, ::-1]
            else: pass

            # Put data in hdu data array
            hdu.data[istart:istop]['data'][:, :, :, f_start:f_stop] = dd[:]

            # Write to file
            hdulist.flush()

	# Close open fil file
        f.close()

    # Close open FITS file
    hdulist.close()
 
    return 


def get_edge_channels(filfiles, chan_per_spw, mask_edge=1):
    # Read in header info
    fmtdict = sigproc.fmtdict
    hd_list = []
    for filfile in filfiles:
        hdi = sigproc.get_header_dict(filfile, 4, fmtdict)
	hd_list.append(hdi)

    hd, err_code = check_headers_and_get_info(hd_list)

    if err_code > 0:
        print("ERROR CODE: %d" %err_code)
        return
    else: pass

    df        = hd['df']
    freq_lo   = hd['fmin']
    freq_hi   = hd['fmax']
    nchan_vals  = hd['nchans']

    nchans = int( (freq_hi - freq_lo) / df ) + 1
    freqs = freq_lo + np.arange(nchans) * df 

    # Find start channels
    fstart_bins = []
    fstop_bins = []
    for ii, hd in enumerate(hd_list):
       fmin, fmax = get_min_max_freqs(hd)
       fstart = int( (fmin - freq_lo) / df )
       fstop  = fstart + nchan_vals[ii] 
       fstart_bins.append(fstart)
       fstop_bins.append(fstop)

    out_mask = np.array([])
    for ii, hd in enumerate(hd_list):
       wts = np.ones(nchan_vals[ii])
       wts = np.reshape(wts, (-1, chan_per_spw))
       wts[:, :mask_edge] = 0
       wts[:, -1 * mask_edge:] = 0 
       wts = wts.ravel()
       mask_xx = np.where(wts == 0)[0]
       mask_idx = fstart_bins[ii] + mask_xx
       mask_idx = mask_idx[ mask_idx <= fstop_bins[ii] ]
       out_mask = np.hstack( (out_mask, mask_idx) ) 

    out_mask = np.unique(out_mask).astype('int')
    
    return out_mask


def calc_stats_from_fits(fitsfile):
    hdulist = fits.open(fitsfile)
    hdu = hdulist[1]
    Nsub = hdu.data['data'].shape[0]
    n_read_subints = 50 

    Nt = 0
    sum_d = None
    sum_d2 = None 

    for istart in np.arange(0, Nsub, n_read_subints):
       print("%d/%d" %(istart, Nsub))
       istop = istart + n_read_subints
       if istop > Nsub:
          istop = Nsub
       else:
          pass
       isub = istop - istart  

       dd = hdu.data[istart : istop]['data']
       ntsamps = dd.shape[0] * dd.shape[1]
       nifs = dd.shape[2]
       nfreqs = dd.shape[3]
       dd = np.reshape(dd, (ntsamps, nifs, nfreqs))
       
       if sum_d is None:
          sum_d = np.sum(dd, axis=0)
          sum_d2 = np.sum(dd**2.0, axis=0)
       else:
          sum_d += np.sum(dd, axis=0)
          sum_d2 += np.sum(dd**2.0, axis=0)

       Nt += ntsamps 

    hdulist.close()
    
    dmean = sum_d / float(Nt)
    dsig  = np.sqrt(Nt * sum_d2 - sum_d**2.0) / float(Nt)
    return dmean, dsig


def meansub_and_mask(fitsfile, means, mask):
    # Open PSRFITS file 
    hdulist = fits.open(fitsfile, mode='update')
    hdu = hdulist[1]
    Nsub = len(hdu.data[:]['data'])
    Nsub = 20
    # Loop through chunks of data to write to PSRFITS
    n_read_subints = 10
    for istart in np.arange(0, Nsub, n_read_subints):
        print("%d/%d" %(istart, Nsub))
        istop = istart + n_read_subints
        if istop > Nsub:
            istop = Nsub
        else:
            pass
        isub = istop - istart  
        
        # Put data in hdu data array
        dd = hdu.data[istart:istop]['data']
	dd -= means
	dd *= mask
        # Write to file
        hdulist.flush()

    hdulist.close()
    return 


def bandpass_and_mask(fitsfile, means, maskchans):
    # Open PSRFITS file 
    hdulist = fits.open(fitsfile, mode='update')
    hdu = hdulist[1]
    Nsub, npblk, nifs, nchans = hdu.data[:]['data'].shape
    #Nsub = 20

    # Turn means to scales
    # Note -- means should have shape (Nifs, Nchans)
    rescale_val = 16.0
    xx = np.where(means <= 0)
    scales = rescale_val / means
    if len(xx):
       scales[xx] = 0
    else:
       pass

    # Apply bitmask to scales
    scales[:, maskchans] = 0 

    # Loop through chunks of data to write to PSRFITS
    n_read_subints = 50
    for istart in np.arange(0, Nsub, n_read_subints):
        print("%d/%d" %(istart, Nsub))
        istop = istart + n_read_subints
        if istop > Nsub:
            istop = Nsub
        else:
            pass
        isub = istop - istart  
        
        # Put data in hdu data array
        dd = hdu.data[istart:istop]['data'].astype('float')
	dd *= scales 
        hdu.data[istart:istop]['data'] = dd.astype('uint8')
         
        # Write to file
        hdulist.flush()

    hdulist.close()
    return 


def fil2fits_full(filfiles, outname, spw_chan, mask_edge=1, 
                  npsub=-1, maxpol=4, fill_fits=True, calc_stat=True, 
                  bpass=True):
    mask = get_edge_channels(filfiles, spw_chan, mask_edge=mask_edge)
    fitsfile = "%s.fits" %(outname)
    
    if fill_fits:
        combine_convert_fil2psrfits(filfiles, outname, npsub=npsub, 
                                    maxpol=maxpol)
    else: 
	# Make sure file exists
        if not os.path.isfile(fitsfile):
            print("%s not found!" %fitsfile)
            return
    
    if calc_stat:
        dmean, dsig = calc_stats_from_fits(fitsfile) 
        np.save("%s_mean.npy" %(outname), dmean)
        np.save("%s_sig.npy" %(outname), dsig)
    else: pass

    if bpass:
        dmean = np.load("%s_mean.npy" %(outname))
        dsig  = np.load("%s_sig.npy" %(outname))
        bandpass_and_mask(fitsfile, dmean, mask)
    return  


def multi_fil2fits(scans, outbase, fil_dir, spw_chan, mask_edge=1,
                  npsub=-1, maxpol=4, fill_fits=True, 
                  calc_stat=True, bpass=True):
    for scan in scans:
	glob_str = "%s/17B-283*.%d.1.[AB][CD].fil" %(fil_dir, scan)
	filfiles = glob.glob(glob_str)
	print(filfiles)
	if len(filfiles) == 0:
	    print("NO FIL FILES FOUND!")
	    return
	else: pass
	
	outname = "%s_%d" %(outbase, scan)
	fil2fits_full(filfiles, outname, spw_chan, mask_edge=mask_edge, 
                  npsub=npsub, maxpol=maxpol, fill_fits=fill_fits, 
		  calc_stat=calc_stat, bpass=bpass)
    return

if __name__ == "__main__":
    
    pass

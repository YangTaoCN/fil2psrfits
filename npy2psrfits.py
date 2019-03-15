import numpy as np
from astropy.io import fits
import astropy.time as atime
import astropy.coordinates as acoord
import astropy.units as aunits
import fill_headers as fillhdr

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

def convert_npy_to_psrfits(npyfile, outname, src_name="", 
                           ra_str="00:00:00", dec_str="00:00:00",
                           mjd_start=0.0, dt=0.001, freq_lo=1400.0, chan_df=1.0,
                           beam_info=np.array([0.0, 0.0, 0.0]), 
                           chan_weights=None, tgrow_zeros=0, 
                           skip_start_zeros=False):
    # Load data (take real part)
    print(" Loading file: %s" %npyfile)
    data = np.real(np.load(npyfile)).astype(np.float32)

    # Obs Specific Metadata
    # Time Info
    mjd      = mjd_start
    dt       = dt  # seconds
    print("  MJD START: %.8f  " %mjd_start)
    print("  TIME RES : %.3f ms" %(dt * 1000.0))
    # Frequency Info (All freqs in MHz)
    nchan    = data.shape[1]
    chan_bw  = chan_df
    bw       = nchan * chan_bw
    fcenter  = freq_lo + 0.5 * (nchan - 1) * chan_df
    freqs    = np.arange(nchan) * chan_df + freq_lo 
    print("  nchan    : %d" %nchan)
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
    # Datum Size (Hard-coded for now)
    nbits    = 32
    print(" DATA WILL BE WRITTEN AS 32BIT FLOATS")
    # Output file name
    outfile = "%s.fits" %outname
    print(" OUTPUT FILE NAME: %s" %outfile)
    
    # Fill in the ObsInfo class
    d = ObsInfo()
    d.fill_from_mjd(mjd)
    d.fill_freq_info(fcenter, nchan, chan_bw)
    d.fill_source_info(src_name, ra_str, dec_str)
    d.fill_beam_info(bmaj_deg, bmin_deg, bpa_deg)
    d.fill_data_info(dt, nbits)
    d.calc_start_lst(mjd)

    # Determine subint size for PSRFITS table
    n_per_subint = int(1.0 / dt)  # Want a subint of about 1s
    n_subints    = int(data.shape[0] / n_per_subint)
    t_subint     = n_per_subint * dt
    d.nsblk    = n_per_subint
    d.scan_len = t_subint * n_subints

    # Reshape data array
    data = data[: n_per_subint * n_subints]
    data = data.reshape( (n_subints, n_per_subint * nchan) )
    
    # Find out if rows have any non-zero data
    # 1 = non-zero data, 0 = all zeros
    nonzero_row = np.any(data, axis=1) * 1.0

    # Expand zeros if so desired
    if tgrow_zeros:
        xx = np.where(nonzero_row == 0)[0]
        dtz = int(tgrow_zeros)
        if len(xx):
            for xxi in xx:
                sl = slice(max(xxi-dtz, 0), min(xxi+dtz, data.shape[0]))
                nonzero_row[sl] = 0
        else: pass
    else: pass

    tstart = 0.0
    # Skip start zero rows (if desired)
    if skip_start_zeros:
        xx_start = np.min(np.where(nonzero_row))
        data = data[xx_start:]
        nonzero_row = nonzero_row[xx_start:]
        n_subints -= xx_start
        d.scan_len = t_subint * n_subints
        tstart = xx_start * t_subint
    else:
        pass    

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

    # Add user-input weights if they make sense
    if chan_weights is not None and len(chan_weights) == nchan:
        dat_wts = np.vstack( [chan_weights] * n_subints ).astype(np.float32)
    else:
        dat_wts  = np.ones(  (n_subints, nchan), dtype=np.float32 )

    # zero-weight rows with no nozero data
    dat_wts *= nonzero_row.reshape((-1,1))
        
    dat_offs = np.zeros( (n_subints, nchan), dtype=np.float32 )
    dat_scl  = np.ones(  (n_subints, nchan), dtype=np.float32 )

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
        fits.Column(name="DAT_FREQ", format='%dE'%nchan, unit='MHz', array=dat_freq),
        fits.Column(name="DAT_WTS" , format='%dE'%nchan, array=dat_wts),
        fits.Column(name="DAT_OFFS", format='%dE'%nchan, array=dat_offs),
        fits.Column(name="DAT_SCL" , format='%dE'%nchan, array=dat_scl),
        fits.Column(name="DATA"    , format=str(nchan*n_per_subint) + 'E', 
                    dim='(%d,1,%d)' %(nchan, n_per_subint), array=data),
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


if __name__ == "__main__":
    
    pass

    """
    beam_num = 1

    beam_locs = [['J2000', '05:31:58.000001', '+33:08:04.00001'],
                 ['J2000', '05:32:45.150000', '+33:01:01.42000']]
    
    npy_dir  = "/lustre/rwharton/15B-378/processing"
    basename = "nov25_run1"
    
    npyfile = "%s/%s_beam%02d.npy" %(npy_dir, basename, beam_num)
    outname = "%s_beam%02d" %(basename, beam_num)

    kwarg_params = {"src_name" : "NOV25_BEAM%02d" %beam_num, 
                    "ra_str"   : beam_locs[beam_num][1],
                    "dec_str"  : beam_locs[beam_num][2],
                    "mjd_start": 57351.14961866319,
                    "dt"       : 0.005,
                    "freq_lo"  : 1372.0, 
                    "chan_df"  : 1.0,
                    "beam_info": np.array([74.4, 44.5, -72.0]) }

    
    convert_npy_to_psrfits(npyfile, outname, **kwarg_params)
    """
